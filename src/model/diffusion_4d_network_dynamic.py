"""Score network module - Simplified for pure seq->structure folding."""
import torch
import math
from torch import nn
from torch.nn import functional as F
from openfold.utils import feats
from src.data import utils as du
from src.data import all_atom
from src.data import all_atom
from src.model import diffusion_4d_ipa_pytorch_dynamic
import functools as fn
from openfold.utils.tensor_utils import batched_gather
from typing import Dict, Text, Tuple

import torch
from src.model.utils import get_timestep_embedding
from openfold.np import residue_constants as rc
Tensor = torch.Tensor


class DFOLDv2_Embeder(nn.Module):
    """Simple embeder for single-frame structure prediction."""

    def __init__(self, model_conf):
        super(DFOLDv2_Embeder, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed
        
        # Time step embedding
        node_embed_size = self._model_conf.node_embed_size
        edge_embed_size = self._model_conf.edge_embed_size
        time_embed_size = node_embed_size
        
        self.timestep_embed = fn.partial(
            get_timestep_embedding,
            embedding_dim=time_embed_size,
        )

        self.node_timestep_proj = nn.Sequential(
            nn.Linear(time_embed_size, node_embed_size // 2),
            nn.SiLU(),
            nn.Linear(node_embed_size // 2, node_embed_size),
        )
        self.node_ln = nn.LayerNorm(node_embed_size)

        self.edge_timestep_proj = nn.Sequential(
            nn.Linear(time_embed_size, edge_embed_size // 2),
            nn.SiLU(),
            nn.Linear(edge_embed_size // 2, edge_embed_size),
        )
        self.edge_ln = nn.LayerNorm(edge_embed_size)

    def forward(self, node_repr, edge_repr, seq_idx, t):
        """Embeds node and edge features with timestep.
        
        Args:
            node_repr: [1, N, D_node] node features from embeder
            edge_repr: [1, N, N, D_edge] edge features
            seq_idx: [1, N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].

        Returns:
            node_embed: [1, N, D_node]
            edge_embed: [1, N, N, D_edge]
        """
        num_batch, num_res = seq_idx.shape
        t_embed = self.timestep_embed(t)
        
        # Processing node
        node_t_step_embedings = self.node_timestep_proj(t_embed)
        node_t_step_embedings = node_t_step_embedings[:, None, :].expand(num_batch, num_res, -1)

        node_embed = self.node_ln(node_repr)
        node_embed = node_embed + node_t_step_embedings

        # Processing edge
        edge_embed = edge_repr.reshape(num_batch, num_res * num_res, -1)
        edge_t_step_embedings = self.edge_timestep_proj(t_embed)
        edge_t_step_embedings = edge_t_step_embedings[:, None, :].expand(num_batch, num_res * num_res, -1)

        edge_embed = self.edge_ln(edge_embed)
        edge_embed = edge_embed + edge_t_step_embedings
        edge_embed = edge_embed.reshape(num_batch, num_res, num_res, -1)

        return node_embed, edge_embed


class FullScoreNetwork(nn.Module):
    """Full score network for pure seq->structure folding."""

    def __init__(self, model_conf, diffuser):
        super(FullScoreNetwork, self).__init__()
        self._model_conf = model_conf
        self.embedding_layer = DFOLDv2_Embeder(model_conf)

        self.diffuser = diffuser
        self.score_model = diffusion_4d_ipa_pytorch_dynamic.DFOLDIpaScore(model_conf, diffuser)
        self.expand_node = nn.Linear(model_conf.node_input_embed_size, model_conf.node_embed_size)
        self.expand_edge = nn.Linear(128, model_conf.edge_embed_size)
        
    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def forward(self, input_feats, drop_ref=False, is_training=True):
        """Forward computes the reverse diffusion conditionals p(X^t|X^{t+1})
        for each item in the batch.

        Args:
            input_feats: Dictionary containing:
                - node_repr: [N, C_node_in] node features
                - edge_repr: [N, N, 128] edge features
                - rigids_t: [1, N, 7] noised rigids
                - t: diffusion timestep
                - res_mask: [1, N] residue mask
                - fixed_mask: [1, N] fixed mask
                - seq_idx: [1, N] sequence indices
                - aatype: [N] amino acid types
                - torsion_angles_sin_cos: [N, 7, 2] torsion angles
                - is_rna: optional RNA mask

        Returns:
            pred_out: dictionary of model outputs.
        """
        fixed_mask = input_feats['fixed_mask'].type(torch.float32)

        num_res = input_feats['node_repr'].shape[0]

        node_repr = input_feats['node_repr']   # (N, C_node_in)
        edge_repr = input_feats['edge_repr']   # (N, N, 128)

        # Expand node and edge representations
        input_feats['expand_node_repr'] = self.expand_node(node_repr)
        input_feats['expand_edge_repr'] = self.expand_edge(
            edge_repr.reshape(num_res * num_res, -1)
        ).reshape(num_res, num_res, -1)
         
        # Initial embeddings with timestep
        # Use single frame (batch_size=1)
        init_node_embed, init_edge_embed = self.embedding_layer(
            node_repr=input_feats['expand_node_repr'].unsqueeze(0),  # [1, N, C]
            edge_repr=input_feats['expand_edge_repr'].unsqueeze(0),   # [1, N, N, C]
            seq_idx=input_feats['seq_idx'],  # [1, N]
            t=input_feats['t'],
        )
            
        # Run main score network (single frame)
        model_out = self.score_model(
            init_node_embed, 
            init_edge_embed, 
            input_feats,
            drop_ref=drop_ref,
            is_training=is_training
        )
        
        gt_angles = input_feats['torsion_angles_sin_cos']
        angles_pred = self._apply_mask(
            model_out['angles'], 
            gt_angles, 
            1 - fixed_mask[..., None, None]
        )
        unorm_angles = self._apply_mask(
            model_out['unorm_angles'], 
            gt_angles, 
            1 - fixed_mask[..., None, None]
        )
        
        pred_out = {
            'angles': angles_pred,
            'unorm_angles': unorm_angles,
            'rot_score': model_out['rot_score'],
            'trans_score': model_out['trans_score'],
        }
        rigids_pred = model_out['final_rigids']
        
        pred_out['rigids'] = rigids_pred.to_tensor_7()
        
        if is_training:
            # Extended atom37 tables now support RNA (indices 21-25), so we can
            # use compute_backbone_atom37 for both protein and protein-RNA structures.
            atom37_pos, atom37_mask, _, _ = all_atom.compute_backbone_atom37(
                rigids_pred, input_feats['aatype'], angles_pred
            )
            all_frames = feats.torsion_angles_to_frames(
                rigids_pred,
                angles_pred,  
                input_feats['aatype'],
                all_atom.DEFAULT_FRAMES.to(angles_pred.device)
            )
            atom14_pos = all_atom.frames_to_atom14_pos(all_frames, input_feats['aatype'])

            pred_out['atom37'] = atom37_pos.to(rigids_pred.device)
            pred_out['atom14'] = atom14_pos.to(rigids_pred.device)
            pred_out['atom_mask'] = atom37_mask.to(rigids_pred.device)
            pred_out['rigid_update'] = model_out['rigid_update']
            
        return pred_out


def get_rc_tensor(rc_np, aatype):
    return torch.tensor(rc_np, device=aatype.device)[aatype]


def atom14_to_atom37(
    atom14_data: torch.Tensor,
    aatype: torch.Tensor
) -> Tuple:
    """Convert atom14 to atom37 representation."""
    idx_atom37_to_atom14 = get_rc_tensor(rc.RESTYPE_ATOM37_TO_ATOM14, aatype).long()
    no_batch_dims = len(aatype.shape) - 1
    atom37_data = batched_gather(
        atom14_data, 
        idx_atom37_to_atom14, 
        dim=no_batch_dims + 1, 
        no_batch_dims=no_batch_dims + 1
    )
    atom37_mask = get_rc_tensor(rc.RESTYPE_ATOM37_MASK, aatype)
    if len(atom14_data.shape) == no_batch_dims + 2:
        atom37_data *= atom37_mask
    elif len(atom14_data.shape) == no_batch_dims + 3:
        atom37_data *= atom37_mask[..., None].to(dtype=atom37_data.dtype)
    else:
        raise ValueError("Incorrectly shaped data")
    return atom37_data, atom37_mask
