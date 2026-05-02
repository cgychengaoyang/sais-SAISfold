# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
EngramMoEProtenix - Protenix model with EngramMoE replacing PairformerStack.

This model replaces the PairformerStack with DeepSeekPairStackWithSingles
while keeping all other Protenix components intact.
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from src.model.protenix.model.generator import (
    InferenceNoiseScheduler,
    TrainingNoiseSampler,
    sample_diffusion,
    sample_diffusion_training,
)
from src.model.protenix.model.modules.confidence import ConfidenceHead
from src.model.protenix.model.modules.diffusion import DiffusionModule
from src.model.protenix.model.modules.embedders import (
    ConstraintEmbedder,
    InputFeatureEmbedder,
    RelativePositionEncoding,
)
from src.model.protenix.model.modules.head import DistogramHead
from src.model.protenix.model.modules.pairformer import MSAModule, TemplateEmbedder
from src.model.protenix.model.modules.primitives import LinearNoBias
from src.model.protenix.model.sample_confidence import compute_contact_prob
from src.model.protenix.model.utils import simple_merge_dict_list
from src.model.protenix.openfold_local.model.primitives import LayerNorm
from src.model.protenix.utils.logger import get_logger

from .modules.engram_moe import DeepSeekPairStackWithSingles

logger = get_logger(__name__)


class EngramMoEProtenix(nn.Module):
    """
    Protenix model with EngramMoE integration.
    
    Replaces PairformerStack with DeepSeekPairStackWithSingles which includes:
    - Hash-based n-gram memory (Engram)
    - Mixture of Experts (MoE)
    - DeepSeek-style attention blocks
    
    All other components (InputEmbedder, DiffusionModule, etc.) remain unchanged.
    """
    
    def __init__(self, configs) -> None:
        super(EngramMoEProtenix, self).__init__()
        self.configs = configs
        
        # Enable TF32 if configured
        torch.backends.cuda.matmul.allow_tf32 = getattr(configs, 'enable_tf32', False)
        
        # Constants
        self.enable_diffusion_shared_vars_cache = configs.enable_diffusion_shared_vars_cache
        self.enable_efficient_fusion = configs.enable_efficient_fusion
        self.N_cycle = configs.model.N_cycle
        self.N_model_seed = configs.model.N_model_seed
        self.train_confidence_only = configs.train_confidence_only
        
        if self.train_confidence_only:
            assert configs.loss.weight.alpha_diffusion == 0.0
            assert configs.loss.weight.alpha_distogram == 0.0
        
        # Diffusion schedulers (use Protenix's exactly)
        self.train_noise_sampler = TrainingNoiseSampler(**configs.train_noise_sampler)
        self.inference_noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )
        self.diffusion_batch_size = configs.diffusion_batch_size
        
        # Model components (standard Protenix)
        esm_configs = configs.get("esm", {})
        self.input_embedder = InputFeatureEmbedder(
            **configs.model.input_embedder, esm_configs=esm_configs
        )
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        self.template_embedder = TemplateEmbedder(**configs.model.template_embedder)
        self.msa_module = MSAModule(
            **configs.model.msa_module,
            msa_configs=configs.data.get("msa", {}),
        )
        # Constraint embedder (optional, only if constraints enabled)
        constraint_config = configs.model.constraint_embedder
        if any([
            constraint_config.get('pocket_embedder', {}).get('enable', False),
            constraint_config.get('contact_embedder', {}).get('enable', False),
            constraint_config.get('contact_atom_embedder', {}).get('enable', False),
            constraint_config.get('substructure_embedder', {}).get('enable', False),
        ]):
            self.constraint_embedder = ConstraintEmbedder(
                **constraint_config
            )
        else:
            self.constraint_embedder = None
        
        # Replace PairformerStack with EngramMoE version
        engram_config = configs.get('engram_moe', {})
        self.use_engram_moe = configs.get('use_engram_moe', True)
        
        if self.use_engram_moe:
            logger.info("Using EngramMoE PairStack")
            self.engram_moe_stack = DeepSeekPairStackWithSingles(
                c_s=configs.c_s,
                c_z=configs.c_z,
                n_blocks=engram_config.get('n_blocks', 8),
                num_heads=engram_config.get('num_heads', 4),
                engram_layers=engram_config.get('engram_layers', [0, 2, 4, 6]),
                moe_layers=engram_config.get('moe_layers', [0, 2, 4, 6]),
                engram_config={
                    'table_size': engram_config.get('engram_table_size', 5000),
                    'ngram_orders': engram_config.get('ngram_sizes', (3, 4, 5)),
                    'num_hash_heads': engram_config.get('heads_per_ngram', 2),
                },
                moe_config={
                    'num_routed_experts': engram_config.get('num_routed_experts', 64),
                    'num_shared_experts': engram_config.get('num_shared_experts', 2),
                    'top_k': engram_config.get('top_k', 2),
                    'expert_dim': engram_config.get('expert_dim', 256),
                },
                dropout=engram_config.get('dropout', 0.0),
                use_gradient_checkpointing=engram_config.get('use_gradient_checkpointing', True),
            )
        else:
            # Fallback to standard PairformerStack
            from src.model.protenix.model.modules.pairformer import PairformerStack
            logger.info("Using standard PairformerStack")
            self.engram_moe_stack = PairformerStack(**configs.model.pairformer)
        
        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
        self.distogram_head = DistogramHead(**configs.model.distogram_head)
        self.confidence_head = ConfidenceHead(**configs.model.confidence_head)
        
        # Dimensions
        self.c_s, self.c_z, self.c_s_inputs = (
            configs.c_s,
            configs.c_z,
            configs.c_s_inputs,
        )
        
        # Linear projections
        self.linear_no_bias_sinit = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_zinit1 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_zinit2 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_token_bond = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s = LayerNorm(self.c_s)
        
        # Zero init for recycling layers
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)
        
        # MoE auxiliary loss weight
        self.moe_aux_loss_weight = configs.get('moe_aux_loss_weight', 0.01)
    
    def get_pairformer_output(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, ...]:
        """
        Forward pass from input to pairformer (EngramMoE) output.
        
        Returns:
            s_inputs, s, z, moe_aux_loss
        """
        if self.train_confidence_only:
            self.input_embedder.eval()
            self.template_embedder.eval()
            self.msa_module.eval()
            self.engram_moe_stack.eval()
        
        # Input embedding
        s_inputs = self.input_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=chunk_size
        )  # [B, N, 449]
        
        z_constraint = None
        if self.constraint_embedder is not None and "constraint_feature" in input_feature_dict:
            z_constraint = self.constraint_embedder(
                input_feature_dict["constraint_feature"]
            )
        
        # Initialize representations
        s_init = self.linear_no_bias_sinit(s_inputs)  # [B, N, c_s]
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  # [B, N, N, c_z]
        
        if inplace_safe:
            z_init += self.relative_position_encoding(input_feature_dict["relp"])
            z_init += self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
            if z_constraint is not None:
                z_init += z_constraint
        else:
            z_init = z_init + self.relative_position_encoding(
                input_feature_dict["relp"]
            )
            z_init = z_init + self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
            if z_constraint is not None:
                z_init = z_init + z_constraint
        
        # Initialize recycling
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)
        
        # Recycling loop
        total_moe_aux_loss = torch.tensor(0.0, device=s_inputs.device)
        
        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(
                self.training
                and (not self.train_confidence_only)
                and cycle_no == (N_cycle - 1)
            ):
                z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                
                if inplace_safe:
                    if self.template_embedder.n_blocks > 0:
                        z += self.template_embedder(
                            input_feature_dict,
                            z,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                else:
                    if self.template_embedder.n_blocks > 0:
                        z = z + self.template_embedder(
                            input_feature_dict,
                            z,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
                
                # Get sequence tokens for Engram
                seq_tokens = input_feature_dict.get("token_index", None)
                
                # Use EngramMoE stack instead of PairformerStack
                if self.use_engram_moe:
                    s, z, moe_aux_loss = self.engram_moe_stack(
                        s=s,
                        z=z,
                        seq_tokens=seq_tokens,
                        pair_mask=None,
                        single_mask=None,
                    )
                    total_moe_aux_loss = total_moe_aux_loss + moe_aux_loss
                else:
                    s, z = self.engram_moe_stack(
                        s,
                        z,
                        pair_mask=None,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
        
        if self.train_confidence_only:
            self.input_embedder.train()
            self.template_embedder.train()
            self.msa_module.train()
            self.engram_moe_stack.train()
        
        return s_inputs, s, z, total_moe_aux_loss
    
    def sample_diffusion(self, **kwargs) -> torch.Tensor:
        """Sample diffusion process."""
        from src.model.protenix.utils.torch_utils import autocasting_disable_decorator
        
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            ]
        }
        _configs.update(
            {
                "attn_chunk_size": (
                    self.configs.infer_setting.chunk_size if not self.training else None
                ),
                "diffusion_chunk_size": (
                    self.configs.infer_setting.sample_diffusion_chunk_size
                    if not self.training
                    else None
                ),
            }
        )
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion
        )(**_configs, **kwargs)
    
    def run_confidence_head(self, *args, **kwargs):
        """Run confidence head."""
        from src.model.protenix.utils.torch_utils import autocasting_disable_decorator
        
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(
            self.confidence_head
        )(*args, **kwargs)
    
    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        label_full_dict: Optional[dict] = None,
        mode: str = "train",
        current_step: Optional[int] = None,
        symmetric_permutation=None,
    ) -> tuple[dict, dict, dict]:
        """
        Main forward pass.
        
        Returns:
            pred_dict: Predictions
            label_dict: Labels
            log_dict: Logging information
        """
        import time
        
        step_st = time.time()
        N_token = input_feature_dict["token_index"].shape[-1]
        
        log_dict = {}
        pred_dict = {}
        time_tracker = {}
        
        # Get trunk output with EngramMoE
        s_inputs, s, z, moe_aux_loss = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=self.N_cycle,
            inplace_safe=True,
            chunk_size=None,
        )
        
        # Log MoE auxiliary loss
        if moe_aux_loss.item() > 0:
            log_dict["moe_aux_loss"] = moe_aux_loss.item()
        
        step_trunk = time.time()
        time_tracker.update({"pairformer": step_trunk - step_st})
        
        # Sample diffusion
        N_sample = self.configs.sample_diffusion["N_sample"]
        N_step = self.configs.sample_diffusion["N_step"]
        
        noise_schedule = self.inference_noise_scheduler(
            N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype
        )
        
        cache = dict()
        if self.enable_diffusion_shared_vars_cache:
            cache["pair_z"] = self.diffusion_module.diffusion_conditioning.prepare_cache(
                input_feature_dict["relp"], z, False
            )
            cache["p_lm/c_l"] = self.diffusion_module.atom_attention_encoder.prepare_cache(
                input_feature_dict["ref_pos"],
                input_feature_dict["ref_charge"],
                input_feature_dict["ref_mask"],
                input_feature_dict["ref_element"],
                input_feature_dict["ref_atom_name_chars"],
                input_feature_dict["atom_to_token_idx"],
                input_feature_dict["d_lm"],
                input_feature_dict["v_lm"],
                input_feature_dict["pad_info"],
                "",
                cache["pair_z"],
                False,
            )
        else:
            cache["pair_z"] = None
            cache["p_lm/c_l"] = [None, None]
        
        pred_dict["coordinate"] = self.sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=None if cache["pair_z"] is not None else z,
            pair_z=cache["pair_z"],
            p_lm=cache["p_lm/c_l"][0],
            c_l=cache["p_lm/c_l"][1],
            N_sample=N_sample,
            noise_schedule=noise_schedule,
            inplace_safe=True,
            enable_efficient_fusion=self.enable_efficient_fusion,
        )
        
        step_diffusion = time.time()
        time_tracker.update({"diffusion": step_diffusion - step_trunk})
        
        # Contact probabilities
        pred_dict["contact_probs"] = compute_contact_prob(
            distogram_logits=self.distogram_head(z),
            **self._get_bin_params(),
        )
        
        # Confidence logits
        confidence_dict = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s=s,
            z=z,
            pred_coords=pred_dict["coordinate"],
            inplace_safe=True,
        )
        pred_dict.update(confidence_dict)
        
        step_confidence = time.time()
        time_tracker.update({"confidence": step_confidence - step_diffusion})
        
        log_dict.update(time_tracker)
        
        return pred_dict, label_dict, log_dict
    
    def _get_bin_params(self):
        """Get bin parameters for distogram."""
        return {
            "min_bin": self.configs.loss.distogram.min_bin,
            "max_bin": self.configs.loss.distogram.max_bin,
            "no_bins": self.configs.no_bins,
        }
