import sys
import os
import copy
import gc
import logging
import pickle
import random
import time
from collections import defaultdict
from datetime import datetime
import GPUtil

import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import tree
from Bio.SVDSuperimposer import SVDSuperimposer
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
import mdtraj as md

import swanlab
from tqdm import tqdm

from DyneTrion.utils import (
    compute_validation_metrics_all,
    plot_curve_merged,
    plot_rot_trans_curve,
    residue_constants,
)
from openfold.utils import rigid_utils as ru
from openfold.utils.loss import lddt_ca, torsion_angle_loss
from src.data import DyneTrion_data_loader_dynamic


def smooth_lddt_ca(
    pred_ca: torch.Tensor,
    gt_ca: torch.Tensor,
    mask: torch.Tensor,
    cutoff: float = 15.0,
    sigma: float = 1.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute smooth LDDT loss on CA positions.
    
    Args:
        pred_ca: Predicted CA positions, shape (..., N, 3)
        gt_ca: Ground truth CA positions, shape (..., N, 3)
        mask: Mask for valid residues, shape (..., N)
        cutoff: Distance cutoff for inclusion (default 15A)
        sigma: Smoothing parameter for Gaussian (default 1A)
        eps: Small constant for numerical stability
    
    Returns:
        Smooth LDDT score (higher is better), shape (..., N)
    """
    # Compute distance matrices
    pred_dmat = torch.sqrt(
        eps + torch.sum(
            (pred_ca[..., None, :] - pred_ca[..., None, :, :]) ** 2,
            dim=-1,
        )
    )
    gt_dmat = torch.sqrt(
        eps + torch.sum(
            (gt_ca[..., None, :] - gt_ca[..., None, :, :]) ** 2,
            dim=-1,
        )
    )
    
    n = mask.shape[-1]
    device = pred_ca.device
    
    # Mask for valid pairs
    pair_mask = (
        mask[..., None] * mask[..., None, :] *  # Both residues valid
        (gt_dmat < cutoff) *  # Within cutoff
        (1.0 - torch.eye(n, device=device))  # Exclude self
    )
    
    # Distance difference
    dist_diff = torch.abs(pred_dmat - gt_dmat)
    
    # Smooth score using Gaussian: exp(-d^2 / (2*sigma^2))
    # This gives 1.0 when d=0 and smoothly decreases as d increases
    smooth_score = torch.exp(-(dist_diff ** 2) / (2 * sigma ** 2))
    
    # Per-residue LDDT (average over all valid pairs for each residue)
    norm = 1.0 / (eps + pair_mask.sum(dim=-1))
    score = norm * (eps + (pair_mask * smooth_score).sum(dim=-1))
    
    return score


def smooth_lddt_loss(
    pred_ca: torch.Tensor,
    gt_ca: torch.Tensor,
    mask: torch.Tensor,
    cutoff: float = 15.0,
    sigma: float = 1.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Smooth LDDT loss (1 - score) for optimization.
    
    Returns:
        Loss value (lower is better), averaged over batch and residues
    """
    score = smooth_lddt_ca(pred_ca, gt_ca, mask, cutoff, sigma, eps)
    # Convert to loss: 1 - score
    loss = 1.0 - score
    # Average over valid residues
    loss = (loss * mask).sum(dim=-1) / (eps + mask.sum(dim=-1))
    return loss
from src.analysis import utils as au
from src.data import se3_diffuser, all_atom
from src.data import utils as du
from src.experiments import utils as eu
from src.model import diffusion_4d_network_dynamic
from src.toolbox.rot_trans_error import (
    average_quaternion_distances,
    average_translation_distances,
)

class Experiment:

    def __init__(
            self,
            *,
            conf: DictConfig,
        ):
        """Initialize experiment.

        Args:
            exp_cfg: Experiment configuration.
        """
        self._log = logging.getLogger(__name__)
        self._available_gpus = GPUtil.getAvailable(order='memory', limit = 8)
        # Fallback: if no GPUs reported available (e.g., under load), use all GPUs
        if not self._available_gpus:
            self._available_gpus = list(range(torch.cuda.device_count()))

        # Configs
        self._conf = conf
        self._exp_conf = conf.experiment
        if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
            self._exp_conf.name = (f'{self._exp_conf.name}_{HydraConfig.get().job.num}')
        self._diff_conf = conf.diffuser
        self._model_conf = conf.model
        self._data_conf = conf.data
        self._use_recoder = self._exp_conf.use_recoder
        self._use_ddp = self._exp_conf.use_ddp
        self.dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        # 1. initialize ddp info if in ddp mode
        if self._use_ddp :
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            self.ddp_info = eu.get_ddp_info()
            if self.ddp_info['rank'] not in [0,-1]:
                self._log.addHandler(logging.NullHandler())
                self._log.setLevel("ERROR")
                self._use_recoder = False
                self._exp_conf.ckpt_dir = None

        self.trained_epochs = 0
        self.trained_steps = 0

        # Initialize experiment objects
        self._diffuser = se3_diffuser.SE3Diffuser(self._diff_conf)
        self._model = diffusion_4d_network_dynamic.FullScoreNetwork(self._model_conf, self.diffuser)

        num_parameters = sum(p.numel() for p in self._model.parameters())

        # Note: Temporal module training disabled for seq->structure folding model
        # This model does not have temporal components (frame_time=1, ref_number=0, motion_number=0)
        if getattr(self._conf.model.ipa, 'temporal', False) and getattr(self._conf.model.ipa, 'frozen_spatial', False):
            self._log.warning('Temporal training is disabled in this seq->structure model')

        trainable_num_parameters = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        self._exp_conf.num_parameters = num_parameters
        self._exp_conf.trainable_num_parameters  = trainable_num_parameters
        self._log.info(f'Number of model parameters {num_parameters}, trainable parameters:{trainable_num_parameters}')
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._exp_conf.learning_rate, amsgrad=True)
        if conf.experiment.warm_start:
            ckpt_path = conf.experiment.warm_start
            info = self.load_pretrained_model(ckpt_path=ckpt_path)

            # finetune: by default, do NOT reuse step unless explicitly requested
            if info and conf.experiment.reuse_step and ("epoch" in info) and ("step" in info):
                print('Reuse Step')
                self.trained_epochs = int(info["epoch"])
                self.trained_steps = int(info["step"])
                print(f"loading model from: {ckpt_path}")
            else:
                print(f"[warm_start] finetune from: {ckpt_path}")
            # NOTE: finetune usually should NOT load optimizer state; keep it commented
            # if info and ("optimizer" in info) and info["optimizer"] is not None:
            #     self._optimizer.load_state_dict(info["optimizer"])

        self._init_log()
        self._init_best_eval()
        if not self.conf.experiment.training:
            seed = 0
        else:
            seed = dist.get_rank() if self._use_ddp else 0
        self._set_seed(seed)
        self._inference_cache = {}

    def _init_best_eval(self):
        self.best_trained_steps = 0
        self.best_trained_epoch = 0
        self.best_rmse_ca = 10000
        self.best_rmse_all = 10000
        self.best_drmsd = 10000
        self.best_rmsd_ca_aligned = 10000
        self.best_rot_error=1000
        self.best_trans_error = 1000
        self.best_ref_rot_error = 1000
        self.best_ref_trans_error = 1000

    def _init_log(self):

        if self._exp_conf.ckpt_dir is not None:
            # Set-up checkpoint location
            ckpt_dir = os.path.join(
                self._exp_conf.ckpt_dir,
                # self._exp_conf.name,
                self.dt_string )
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
            self._exp_conf.ckpt_dir = ckpt_dir
            self._log.info(f'Checkpoints saved to: {ckpt_dir}')
        else:
            self._log.info('Checkpoint not being saved.')

        if self._exp_conf.eval_dir is not None :
            eval_dir = os.path.join(
                self._exp_conf.eval_dir,
                self._exp_conf.name,
                self.dt_string)
            self._exp_conf.eval_dir = eval_dir
            self._log.info(f'Evaluation saved to: {eval_dir}')
        else:
            self._exp_conf.eval_dir = os.devnull
            self._log.info(f'Evaluation will not be saved.')

    def load_pretrained_model(self, ckpt_path):
        """
        Warm start / finetune loading:
        - Load only matching keys with matching shapes
        - Safely strips "module." prefix from DDP checkpoints
        """
        try:
            self._log.info(f'Loading checkpoint from {ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location='cpu')

            # --- get model state dict ---
            if isinstance(ckpt, dict) and ("model" in ckpt) and isinstance(ckpt["model"], dict):
                state = ckpt["model"]
            elif isinstance(ckpt, dict) and ("state_dict" in ckpt) and isinstance(ckpt["state_dict"], dict):
                state = ckpt["state_dict"]
            elif isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
                state = ckpt
            else:
                raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

            # --- strip DDP prefix safely (only from start of the key) ---
            state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }

            # --- filter by existing keys & matching shapes ---
            model_state = self._model.state_dict()
            filtered = {}
            skipped_shape = []
            for k, v in state.items():
                if k in model_state:
                    if v.shape == model_state[k].shape:
                        filtered[k] = v
                    else:
                        skipped_shape.append(f"{k} ({tuple(v.shape)} vs {tuple(model_state[k].shape)})")

            missing, unexpected = self._model.load_state_dict(filtered, strict=False)

            self._log.info(f'[WarmStart] Successfully loaded {len(filtered)} tensors.')
            if missing:
                self._log.info(f'[WarmStart] {len(missing)} keys missing (expected for new Multimer layers).')
            if skipped_shape:
                self._log.warning(f'[WarmStart] {len(skipped_shape)} tensors skipped due to shape mismatch: {skipped_shape}')

            # return optional meta info
            info = {}
            if isinstance(ckpt, dict):
                for k in ["conf", "optimizer", "epoch", "step"]:
                    if k in ckpt: info[k] = ckpt[k]
            return info

        except Exception as e:
            self._log.error(f"Error loading checkpoint: {e}")
            return None


    @property
    def diffuser(self):
        return self._diffuser

    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf

    def create_dataset(self):

        # Datasets
        train_dataset = DyneTrion_data_loader_dynamic.PdbDataset(
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=True
        )

        valid_dataset = DyneTrion_data_loader_dynamic.PdbDataset(
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=False
        )
        # Loaders
        num_workers = self._exp_conf.num_loader_workers

        persistent_workers = True if num_workers > 0 else False
        prefetch_factor = 2
        prefetch_factor = 2 if num_workers == 0 else prefetch_factor

        sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()) if self._use_ddp else None
        train_loader = data.DataLoader(
                train_dataset,
                batch_size=self._exp_conf.batch_size if not self._exp_conf.use_ddp else self._exp_conf.batch_size // self.ddp_info['world_size'],
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                drop_last=False,
                shuffle=(sampler is None),
                sampler=sampler,
                multiprocessing_context='fork' if num_workers != 0 else None,
        )
        valid_loader = data.DataLoader(
                valid_dataset,
                batch_size=self._exp_conf.eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                drop_last=False,
                multiprocessing_context='fork' if num_workers != 0 else None,
        )

        return train_loader, valid_loader

    def init_swanlab_logger(self):
        self._log.info("Initializing SwanLab Recoder.")
        conf_dict = OmegaConf.to_container(self._conf, resolve=True)
        swanlab_mode = conf_dict["experiment"]["recoder"]["mode"]
        if swanlab_mode == "cloud":
            swanlab.login(api_key=conf_dict["experiment"]["recoder"]["api_key"])
        self.swanlab_logger = swanlab.init(
            project=conf_dict["experiment"]["project"],
            experiment_name=conf_dict["experiment"]["name"],
            config=conf_dict,
            mode=swanlab_mode,
            logdir=conf_dict["experiment"]["recoder"]["save_path"],
        )

    def start_training(self, return_logs=False):
        # Set environment variables for which GPUs to use.
        if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
            replica_id = int(HydraConfig.get().job.num)
        else:
            replica_id = 0
        if self._use_recoder and replica_id == 0:
            self.init_swanlab_logger()
        assert(not self._exp_conf.use_ddp or self._exp_conf.use_gpu)
        # GPU mode
        if torch.cuda.is_available() and self._exp_conf.use_gpu:
            # single GPU mode
            if self._exp_conf.num_gpus==1 :
                gpu_id = self._available_gpus[replica_id]
                device = f"cuda:{gpu_id}"
                self._model = self.model.to(device)
                self._log.info(f"Using device: {device}")
            #muti gpu mode
            elif self._exp_conf.num_gpus > 1:
                device_ids = [f"cuda:{i}" for i in self._available_gpus[:self._exp_conf.num_gpus]]
                #DDP mode
                if self._use_ddp :
                    device = torch.device("cuda",self.ddp_info['local_rank'])
                    model = self.model.to(device)
                    self._model = DDP(model, device_ids=[self.ddp_info['local_rank']], output_device=self.ddp_info['local_rank'],find_unused_parameters=True)
                    self._log.info(f"Multi-GPU training on GPUs in DDP mode, node_id : {self.ddp_info['node_id']}, devices: {device_ids}")
                #DP mode
                else:
                    if len(self._available_gpus) < self._exp_conf.num_gpus:
                        raise ValueError(f"require {self._exp_conf.num_gpus} GPUs, but only {len(self._available_gpus)} GPUs available ")
                    self._log.info(f"Multi-GPU training on GPUs in DP mode: {device_ids}")
                    gpu_id = self._available_gpus[replica_id]
                    device = f"cuda:{gpu_id}"
                    self._model = DP(self._model, device_ids=device_ids)
                    self._model = self.model.to(device)
        else:
            device = 'cpu'
            self._model = self.model.to(device)
            self._log.info(f"Using device: {device}")

        # if self.conf.experiment.warm_start:
        #     for state in self._optimizer.state.values():
        #         for k, v in state.items():
        #             if torch.is_tensor(v):
        #                 state[k] = v.to(device)

        self._model.train()

        (train_loader, valid_loader) = self.create_dataset()

        logs = []
        # torch.cuda.empty_cache()
        for epoch in range(self.trained_epochs, self._exp_conf.num_epoch):
            self.trained_epochs = epoch
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            epoch_log = self.train_epoch(
                train_loader,
                valid_loader,
                device,
                return_logs=return_logs
            )
            # self._schedule.step()

            if return_logs:
                logs.append(epoch_log)
        if self._exp_conf.ckpt_dir is not None:
            ckpt_path = os.path.join(self._exp_conf.ckpt_dir, f'last_step_{self.trained_steps}.pth')
            du.write_checkpoint(
                ckpt_path,
                copy.deepcopy(self.model.state_dict()),
                self._conf,
                copy.deepcopy(self._optimizer.state_dict()),
                self.trained_epochs,
                self.trained_steps,
                logger=self._log,
                use_torch=True
            )
        self._log.info('Done')
        return logs

    def update_fn(self, data):
        """Updates the state using some data and returns metrics."""
        self._optimizer.zero_grad()
        loss, aux_data = self.loss_fn(data)

        # Check for NaN in loss
        if torch.isnan(loss) or torch.isinf(loss):
            self._log.error(f"NaN/Inf detected in loss at step {self.trained_steps}")
            # Check model outputs for NaN
            for key, val in aux_data.items():
                if torch.is_tensor(val) and (torch.isnan(val).any() or torch.isinf(val).any()):
                    self._log.error(f"  NaN/Inf in aux_data['{key}']")
            # Check model parameters for NaN before backward
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    self._log.error(f"  NaN/Inf in model parameter: {name}")
        
        loss.backward()
        
        # Check gradients for NaN
        has_nan_grad = False
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    if not has_nan_grad:
                        self._log.error(f"NaN/Inf detected in gradients at step {self.trained_steps}")
                        has_nan_grad = True
                    self._log.error(f"  NaN/Inf in gradient of: {name}")
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self._optimizer.step()
        
        # Check model parameters for NaN after step
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                self._log.error(f"NaN/Inf in model parameter after step {self.trained_steps}: {name}")

        return loss, aux_data

    def train_epoch(self, train_loader, valid_loader, device,return_logs=False):
        log_lossses = defaultdict(list)
        global_logs = []
        log_time = time.time()
        step_time = time.time()
        # Run evaluation

        for train_feats in train_loader:
            self.model.train()
            train_feats = tree.map_structure(lambda x: x.to(device), train_feats)

            for k, v in list(train_feats.items()):
                if not torch.is_tensor(v):
                    continue
                if v.dim() == 0:
                    continue
                if v.shape[0] != 1:
                    raise ValueError(
                        f"Current training path only supports batch_size == 1, "
                        f"but got key={k}, shape={tuple(v.shape)}"
                    )
                train_feats[k] = v.squeeze(0)

            loss, aux_data = self.update_fn(train_feats)

            if return_logs:
                global_logs.append(loss)
            for k,v in aux_data.items():
                log_lossses[k].append(du.move_to_np(v))
            self.trained_steps += 1

            if self.trained_steps == 1 or self.trained_steps % self._exp_conf.log_freq == 0:
                elapsed_time = time.time() - log_time
                log_time = time.time()
                step_per_sec = self._exp_conf.log_freq / elapsed_time
                rolling_losses = tree.map_structure(np.mean, log_lossses)
                loss_log = ' '.join([
                    f'{k}={v[0]:.4f}'
                    for k,v in rolling_losses.items() if 'batch' not in k
                ])
                self._log.info(f'Epoch[{self.trained_epochs}/{self._exp_conf.num_epoch}] trained_steps: [{self.trained_steps}]: {loss_log}, steps/sec={step_per_sec:.5f}')
                log_lossses = defaultdict(list)

            # Take checkpoint
            if self._exp_conf.ckpt_dir is not None and ((self.trained_steps % self._exp_conf.ckpt_freq) == 0 or (self._exp_conf.early_ckpt and self.trained_steps == 100)):
                ckpt_path = os.path.join(self._exp_conf.ckpt_dir, f'step_{self.trained_steps}.pth')
                du.write_checkpoint(
                    ckpt_path,
                    copy.deepcopy(self.model.state_dict()),
                    self._conf,
                    copy.deepcopy(self._optimizer.state_dict()),
                    self.trained_epochs,
                    self.trained_steps,
                    logger=self._log,
                    use_torch=True
                )

                if self._exp_conf.enable_validation:
                    # Run evaluation
                    self._log.info(f'Running evaluation of {ckpt_path}')
                    start_time = time.time()
                    eval_dir = os.path.join(self._exp_conf.eval_dir, f'step_{self.trained_steps}')
                    os.makedirs(eval_dir, exist_ok=True)
                    results = self.eval_fn(eval_dir, valid_loader, device,
                        noise_scale=self._exp_conf.noise_scale
                    )
                    eval_time = time.time() - start_time
                    eval_logs = {"Eval/Eval_time": float(eval_time)}

                    mean_metrics = results["metrics"].mean(numeric_only=True)
                    for metric_name, value in mean_metrics.items():
                        eval_logs[f"Eval/{metric_name}"] = float(value)
                    # Scalar metrics
                    eval_logs = {
                        "Eval/RigidError_rot_pred": float(results["rot_trans_error_mean"]["ave_rot"]),
                        "Eval/RigidError_trans_pred": float(results["rot_trans_error_mean"]["ave_trans"]),
                    }
                    # Add ref metrics only if available (temporal models)
                    if results["rot_trans_error_mean"]["first_rot"] != 0.0:
                        eval_logs["Eval/RigidError_rot_ref"] = float(results["rot_trans_error_mean"]["first_rot"])
                        eval_logs["Eval/RigidError_trans_ref"] = float(results["rot_trans_error_mean"]["first_trans"])

                    # matplotlib Figure
                    eval_logs.update({
                        "Eval/Fig_dis_curve": swanlab.Image(results["fig_dis_curve"]),
                        "Eval/Fig_dis_curve_aligned": swanlab.Image(results["fig_dis_curve_aligned"]),
                        "Eval/Fig_error": swanlab.Image(results["fig_error"]),
                    })
                    swanlab.log(eval_logs, step=self.trained_steps)

                    info_dict = {
                        "info/best_trained_steps": self.best_trained_steps,
                        "info/best_trained_epoch": self.best_trained_epoch,
                        "info/best_rmse_all": float(self.best_rmse_all),
                        "info/relat_rmse_ca": float(self.best_rmse_ca),
                        "info/rot_error": float(self.best_rot_error),
                        "info/ref_rot_error": float(self.best_ref_rot_error),
                        "info/trans_error": float(self.best_trans_error),
                        "info/ref_trans_error": float(self.best_ref_trans_error),
                        "info/relat_rmsd_ca_aligned": float(self.best_rmsd_ca_aligned),
                        "info/relat_drmsd": float(self.best_drmsd),
                    }
                    swanlab.log(info_dict, print_to_console=True)
                    self._log.info(f'Finished evaluation in {eval_time:.2f}s')

            # Remote log to tensorborad.
            if self._use_recoder:
                step_time = time.time() - step_time
                example_per_sec = self._exp_conf.batch_size / step_time
                step_time = time.time()
                # Logging basic metrics
                log_metrics = {
                    # Losses
                    "Train/Loss_total": loss,
                    "Train/Loss_rot": aux_data["rot_loss"],
                    "Train/Loss_trans": aux_data["trans_loss"],
                    "Train/Loss_torsion": aux_data["torsion_loss"],
                    "Train/Loss_bb_atom": aux_data["bb_atom_loss"],
                    "Train/Loss_dist_mat": aux_data["dist_mat_loss"],
                    "Train/Loss_interface": aux_data["interface_loss"],
                    "Train/Loss_clash": aux_data["clash_loss"],
                    "Train/Loss_chain_com": aux_data["chain_com_loss"],
                    "Train/Loss_smooth_lddt": aux_data["smooth_lddt_loss"],
                    # Rigid updates
                    "Train/Rigid_rot0": aux_data["update_rots"][0],
                    "Train/Rigid_rot1": aux_data["update_rots"][1],
                    "Train/Rigid_rot2": aux_data["update_rots"][2],
                    "Train/Rigid_trans0": aux_data["update_trans"][0],
                    "Train/Rigid_trans1": aux_data["update_trans"][1],
                    "Train/Rigid_trans2": aux_data["update_trans"][2],
                    # Speed
                    "Train/Speed_examples_per_sec": float(example_per_sec),
                }


                bb_grads = [p.grad for name, p in self.model.named_parameters() if p.grad is not None and "bb_update" in name]
                if bb_grads:
                    bb_grads_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2.0) for g in bb_grads]), 2.0).item()
                    log_metrics["Train/Grad_bb_update"] = bb_grads_norm

                # Logging checkpoint metrics if available
                if torch.isnan(loss):
                    swanlab.log({"Alerts": f"Encountered NaN loss after {self.trained_epochs} epochs, {self.trained_steps} steps"},
                                step=self.trained_steps)
                    raise Exception("NaN encountered")

                swanlab.log(log_metrics, step=self.trained_steps)

        if return_logs:
            return global_logs

    def eval_fn(self, eval_dir, valid_loader, device, min_t=None, num_t=None, noise_scale=1.0,is_training=True):
        # === 初始化路径与指标 ===
        dirs = self._prepare_eval_dirs(eval_dir, is_training)
        metrics = self._init_eval_metrics()

        # === Evaluate each sample ===
        for valid_feats, pdb_names, start_index in tqdm(valid_loader, desc="Evaluating", ncols=100):
            result = self._process_one_protein_for_eval(
                valid_feats, pdb_names, start_index,
                device=device,
                dirs=dirs,
                min_t=min_t,
                num_t=num_t,
                noise_scale=noise_scale,
                is_training=is_training
            )
            self._accumulate_metrics(metrics, result)

        # === Generate summary results ===
        return self._finalize_eval_outputs(metrics, dirs, eval_dir)


    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def eval_extension(self, eval_dir, valid_loader, device, min_t=None, num_t=None, noise_scale=1.0,seed=42):
        self._set_seed(seed)
        # ergodic the validation
        pdb_base_path, ref_base_path = self._prepare_extension_eval_dirs(eval_dir)
        print(f"\n{'*' * 10} Protein number: {len(valid_loader)} {'*' * 10}")
        extrapolation_time = self._conf.eval.extrapolation_time

        for valid_feats, pdb_names, start_index  in valid_loader:
            self._process_one_protein_extrapolation(
                extrapolation_time,
                valid_feats,
                pdb_names,
                ref_base_path,
                pdb_base_path,
                device,
                min_t,
                num_t,
                noise_scale,
            )


    def _self_conditioning(self, batch):
        model_sc = self.model(batch)
        batch['sc_ca_t'] = model_sc['rigids'][..., 4:]
        return batch

    def loss_fn(self, batch):
        """Computes loss and auxiliary data.

        Args:
            batch: Batched data.
            model_out: Output of model ran on batch.

        Returns:
            loss: Final training loss scalar.
            aux_data: Additional logging data.
        """

        if self._model_conf.embed.embed_self_conditioning and random.random() > 0.5:
            with torch.no_grad():
                batch = self._self_conditioning(batch)
        model_out = self.model(batch)
        multimer_conf = getattr(self._model_conf, "multimer", None)
        use_multimer = bool(getattr(multimer_conf, "enabled", False))

        bb_mask = batch['res_mask']
        diffuse_mask = 1 - batch['fixed_mask']

        loss_mask = bb_mask * diffuse_mask
        batch_size, num_res = bb_mask.shape

        torsion_loss = torsion_angle_loss(
            a=model_out['angles'],
            a_gt=batch['torsion_angles_sin_cos'],
            a_alt_gt=batch['alt_torsion_angles_sin_cos'],
            mask=batch['torsion_angles_mask']) * self._exp_conf.torsion_loss_weight

        gt_rot_score = batch['rot_score']
        rot_score_scaling = batch['rot_score_scaling']

        # rot_score_scaling can be a scalar or length-1 tensor (shared across frames).
        # But in our training, the leading dimension is treated as batch_size (=T when B==1),
        # so we need rot_score_scaling to be shape (batch_size,).
        if torch.is_tensor(rot_score_scaling):
            if rot_score_scaling.dim() == 0:
                rot_score_scaling = rot_score_scaling.view(1)  # () -> (1,)
            if rot_score_scaling.numel() == 1 and batch_size > 1:
                rot_score_scaling = rot_score_scaling.expand(batch_size)  # (1,) -> (batch_size,)
        batch_loss_mask = torch.any(bb_mask, dim=-1)

        pred_rot_score = model_out['rot_score'] * diffuse_mask[..., None]
        pred_trans_score = model_out['trans_score'] * diffuse_mask[..., None]

        # Translation x0 loss
        gt_trans_x0 = batch['rigids_0'][..., 4:] #* self._exp_conf.coordinate_scaling
        pred_trans_x0 = model_out['rigids'][..., 4:] #* self._exp_conf.coordinate_scaling

        # Note: ref_trans_loss removed - not applicable for single-frame structure prediction
        # (was: temporal consistency loss comparing to reference frame)

        trans_loss = torch.sum(
            (gt_trans_x0 - pred_trans_x0).abs() * loss_mask[..., None],
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)
        trans_loss *= self._exp_conf.trans_loss_weight

        rot_mse = (gt_rot_score - pred_rot_score)**2 * loss_mask[..., None]
        rot_loss = torch.sum(
            rot_mse / rot_score_scaling[:, None, None]**2,
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)
        rot_loss *= self._exp_conf.rot_loss_weight

        # Note: ref_rot_loss removed - not applicable for single-frame structure prediction
        # (was: temporal consistency loss comparing to reference frame)
        rot_loss *= batch['t'] > self._exp_conf.rot_loss_t_threshold

        rot_loss *= int(self._diff_conf.diffuse_rot)

        # Backbone atom loss
        pred_atom37 = model_out['atom37'][:, :, :5]
        gt_rigids = ru.Rigid.from_tensor_7(batch['rigids_0'].type(torch.float32))
        gt_psi = batch['torsion_angles_sin_cos'][..., 2, :]  # psi
        gt_atom37, atom37_mask, _, _ = all_atom.compute_backbone(gt_rigids, gt_psi) # psi

        gt_atom37 = gt_atom37
        atom37_mask = atom37_mask

        gt_atom37 = gt_atom37[:, :, :5]
        atom37_mask = atom37_mask[:, :, :5]

        gt_atom37 = gt_atom37.to(pred_atom37.device)
        atom37_mask = atom37_mask.to(pred_atom37.device)
        bb_atom_loss_mask = atom37_mask * loss_mask[..., None]
        bb_atom_loss = torch.sum(
            (pred_atom37 - gt_atom37)**2 * bb_atom_loss_mask[..., None],
            dim=(-1, -2, -3)
        ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + 1e-10)
        bb_atom_loss *= self._exp_conf.bb_atom_loss_weight
        # TODO here delete the filter
        bb_atom_loss *= batch['t'] < self._exp_conf.bb_atom_loss_t_filter

        bb_atom_loss *= self._exp_conf.aux_loss_weight

        gt_flat_atoms = gt_atom37.reshape([batch_size, num_res*5, 3])
        gt_pair_dists = torch.linalg.norm(gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_atom37.reshape([batch_size, num_res*5, 3])
        pred_pair_dists = torch.linalg.norm(pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 5))
        flat_loss_mask = flat_loss_mask.reshape([batch_size, num_res*5])
        flat_res_mask = torch.tile(bb_mask[:, :, None], (1, 1, 5))
        flat_res_mask = flat_res_mask.reshape([batch_size, num_res*5])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        # No loss on anything >6A
        proximity_mask = gt_pair_dists < 6
        pair_dist_mask  = pair_dist_mask * proximity_mask

        dist_mat_loss = torch.sum((gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,dim=(1, 2))
        dist_mat_denom = (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res).clamp_min(1.0)
        dist_mat_loss /= dist_mat_denom
        dist_mat_loss *= self._exp_conf.dist_mat_loss_weight
        dist_mat_loss *= batch['t'] < self._exp_conf.dist_mat_loss_t_filter
        dist_mat_loss *= self._exp_conf.aux_loss_weight

        # === Multimer: interface loss & chain clash loss (CA-level) ===
        interface_loss = torch.zeros_like(dist_mat_loss)
        clash_loss = torch.zeros_like(dist_mat_loss)

        interface_w = float(getattr(self._exp_conf, "interface_dist_loss_weight", 0.0))
        interface_cut = float(getattr(self._exp_conf, "interface_dist_cutoff", 8.0))
        clash_w = float(getattr(self._exp_conf, "chain_clash_loss_weight", 0.0))
        clash_thr = float(getattr(self._exp_conf, "chain_clash_ca_threshold", 2.0))

        if use_multimer and interface_w > 0.0 and ("asym_id" in batch):
            asym = batch["asym_id"].long()  # (B,N)
            same_chain = (asym[:, :, None] == asym[:, None, :])  # (B,N,N)
            diff_chain = ~same_chain

            # CA positions from rigids translations
            gt_ca = gt_trans_x0  # (B,N,3)
            pred_ca = pred_trans_x0

            gt_d = torch.cdist(gt_ca, gt_ca, p=2)    # (B,N,N)
            pred_d = torch.cdist(pred_ca, pred_ca, p=2)

            # valid residue pairs
            lm = (loss_mask > 0.5)
            pair_mask = (lm[:, :, None] & lm[:, None, :]) & diff_chain
            # interface pairs defined by GT proximity
            iface_mask = pair_mask & (gt_d < interface_cut)

            # exclude diagonal
            eye = torch.eye(num_res, device=gt_d.device).bool().unsqueeze(0)
            iface_mask = iface_mask & (~eye)

            # MSE on distances at the interface
            denom = iface_mask.sum(dim=(1, 2)).clamp_min(1.0)
            interface_loss = ((gt_d - pred_d) ** 2 * iface_mask.float()).sum(dim=(1, 2)) / denom
            interface_loss = interface_loss * interface_w

        if use_multimer and clash_w > 0.0 and ("asym_id" in batch):
            asym = batch["asym_id"].long()
            diff_chain = (asym[:, :, None] != asym[:, None, :])

            pred_ca = pred_trans_x0
            pred_d = torch.cdist(pred_ca, pred_ca, p=2)

            lm = (loss_mask > 0.5)
            pair_mask = (lm[:, :, None] & lm[:, None, :]) & diff_chain
            eye = torch.eye(num_res, device=pred_d.device).bool().unsqueeze(0)
            pair_mask = pair_mask & (~eye)

            # penalty for too-close inter-chain CA distances
            too_close = (pred_d < clash_thr) & pair_mask
            denom = too_close.sum(dim=(1, 2)).clamp_min(1.0)
            clash_loss = ((clash_thr - pred_d).clamp_min(0.0) ** 2 * too_close.float()).sum(dim=(1, 2)) / denom
            clash_loss = clash_loss * clash_w

        # === AF-Multimer-inspired chain center-of-mass loss (prevents chain overlap / collapse) ===
        chain_com_loss = torch.zeros_like(dist_mat_loss)
        com_w = float(getattr(self._exp_conf, "chain_com_loss_weight", 0.0))
        com_margin = float(getattr(self._exp_conf, "chain_com_margin", 1.0))
        if use_multimer and com_w > 0.0 and ("asym_id" in batch):
            asym = batch["asym_id"].long()
            lm = (loss_mask > 0.5)
            pred_ca = pred_trans_x0
            gt_ca = gt_trans_x0
            for b in range(batch_size):
                chain_ids = torch.unique_consecutive(asym[b][lm[b]]) if lm[b].any() else torch.tensor([], device=asym.device, dtype=asym.dtype)
                if chain_ids.numel() < 2:
                    continue
                pred_coms, gt_coms = [], []
                for cid in chain_ids.tolist():
                    cmask = (asym[b] == cid) & lm[b]
                    if cmask.sum() == 0:
                        continue
                    pred_coms.append(pred_ca[b, cmask].mean(dim=0))
                    gt_coms.append(gt_ca[b, cmask].mean(dim=0))
                if len(pred_coms) < 2:
                    continue
                pred_coms = torch.stack(pred_coms, dim=0)
                gt_coms = torch.stack(gt_coms, dim=0)
                pred_cd = torch.cdist(pred_coms, pred_coms)
                gt_cd = torch.cdist(gt_coms, gt_coms)
                k = pred_cd.shape[0]
                eye = torch.eye(k, device=pred_cd.device).bool()
                pair = ~eye
                # only penalize predicted COM becoming much closer than GT (collapse)
                target_min = (gt_cd - com_margin).clamp_min(0.0)
                pen = (target_min - pred_cd).clamp_min(0.0) ** 2
                chain_com_loss[b] = pen[pair].mean() if pair.any() else 0.0
            chain_com_loss = chain_com_loss * com_w

        # Smooth LDDT loss on CA positions
        smooth_lddt_loss_weight = float(getattr(self._exp_conf, "smooth_lddt_loss_weight", 0.0))
        smooth_lddt_loss_val = torch.zeros_like(dist_mat_loss)
        if smooth_lddt_loss_weight > 0.0:
            smooth_lddt_loss_val = smooth_lddt_loss(
                pred_ca=pred_trans_x0,  # (B, N, 3)
                gt_ca=gt_trans_x0,  # (B, N, 3)
                mask=loss_mask,  # (B, N)
                cutoff=15.0,
                sigma=1.0,
            )
            smooth_lddt_loss_val = smooth_lddt_loss_val * smooth_lddt_loss_weight

        batch_loss_mask = batch_loss_mask
        final_loss = (
            rot_loss
            + trans_loss
            + bb_atom_loss
            + dist_mat_loss
            + torsion_loss
            + interface_loss
            + clash_loss
            + chain_com_loss
            + smooth_lddt_loss_val
        )
        def normalize_loss(x):
            return x.sum() /  (batch_loss_mask.sum() + 1e-10)
        aux_data = {
            'batch_train_loss': final_loss.detach(),
            'batch_rot_loss': rot_loss.detach(),
            'batch_trans_loss': trans_loss.detach(),
            'batch_bb_atom_loss': bb_atom_loss.detach(),
            'batch_dist_mat_loss': dist_mat_loss.detach(),
            'batch_interface_loss': interface_loss.detach(),
            'batch_clash_loss': clash_loss.detach(),
            'batch_chain_com_loss': chain_com_loss.detach(),
            'batch_torsion_loss': torsion_loss.detach(),
            'batch_smooth_lddt_loss': smooth_lddt_loss_val.detach(),
            'total_loss': normalize_loss(final_loss).detach(),
            'rot_loss': normalize_loss(rot_loss).detach(),
            'trans_loss': normalize_loss(trans_loss).detach(),
            'bb_atom_loss': normalize_loss(bb_atom_loss).detach(),
            'dist_mat_loss': normalize_loss(dist_mat_loss).detach(),
            'interface_loss': normalize_loss(interface_loss).detach(),
            'clash_loss': normalize_loss(clash_loss).detach(),
            'chain_com_loss': normalize_loss(chain_com_loss).detach(),
            'torsion_loss': normalize_loss(torsion_loss).detach(),
            'smooth_lddt_loss': normalize_loss(smooth_lddt_loss_val).detach(),
            'update_rots': torch.mean(torch.abs(model_out['rigid_update'][...,:3]),dim=(0,1)).detach(),
            'update_trans': torch.mean(torch.abs(model_out['rigid_update'][...,-3:]),dim=(0,1)).detach(),
        }

        assert final_loss.shape == (batch_size,)
        assert batch_loss_mask.shape == (batch_size,)

        return normalize_loss(final_loss), aux_data

    def _calc_trans_0(self, trans_score, trans_t, t):
        beta_t = self._diffuser._se3_diffuser._r3_diffuser.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        cond_var = 1 - torch.exp(-beta_t)
        return (trans_score * cond_var + trans_t) / torch.exp(-1/2*beta_t)

    def _set_t_feats(self, feats, t, t_placeholder):
        feats['t'] = t * t_placeholder
        rot_score_scaling, trans_score_scaling = self.diffuser.score_scaling(t)
        feats['rot_score_scaling'] = rot_score_scaling * t_placeholder
        feats['trans_score_scaling'] = trans_score_scaling * t_placeholder
        return feats

    def _get_inference_cache(self, device, num_t=None, min_t=None):
        num_t = int(self._data_conf.num_t if num_t is None else num_t)
        min_t = float(self._data_conf.min_t if min_t is None else min_t)
        device = torch.device(device)
        cache_key = (device.type, device.index, num_t, min_t)
        cached = self._inference_cache.get(cache_key)
        if cached is not None:
            return cached

        reverse_steps = torch.linspace(min_t, 1.0, num_t, device=device).flip(0)
        t_placeholder = torch.ones((1,), device=device)
        dt = 1.0 / num_t
        sqrt_dt = torch.sqrt(torch.tensor(dt, device=device))

        self.diffuser._so3_diffuser.to_device(device)
        self.diffuser._r3_diffuser.to_device(device)
        with torch.no_grad():
            all_rot_scales, all_trans_scales = self.diffuser.score_scaling(reverse_steps)

        cached = {
            "reverse_steps": reverse_steps,
            "t_placeholder": t_placeholder,
            "dt": dt,
            "sqrt_dt": sqrt_dt,
            "all_rot_scales": all_rot_scales.to(device),
            "all_trans_scales": all_trans_scales.to(device),
            "sc_t": reverse_steps[0],
            "sc_rot_scale": all_rot_scales[0],
            "sc_trans_scale": all_trans_scales[0],
        }
        self._inference_cache[cache_key] = cached
        return cached

    def forward_traj(self, x_0, min_t, num_t):
        forward_steps = np.linspace(min_t, 1.0, num_t)[:-1]
        x_traj = [x_0]
        for t in forward_steps:
            x_t = self.diffuser.se3_diffuser._r3_diffuser.forward(
                x_traj[-1], t, num_t)
            x_traj.append(x_t)
        x_traj = torch.stack(x_traj, axis=0)
        return x_traj

    def inference_fn(
            self,
            data_init,
            num_t=None,
            min_t=None,
            center=True,
            aux_traj=False,
            self_condition=True,
            noise_scale=1.0,
            z_rot_all=None,
            z_trans_all=None,
        ):
        """Inference function.

        Args:
            data_init: Initial data values for sampling.
        """
        self._model.eval()
        sample_feats = {k: v for k, v in data_init.items()}
        device = sample_feats['rigids_t'].device
        cache = self._get_inference_cache(device, num_t=num_t, min_t=min_t)
        reverse_steps = cache["reverse_steps"]
        t_placeholder = cache["t_placeholder"]
        dt = cache["dt"]
        sqrt_dt = cache["sqrt_dt"]
        all_rot_scales = cache["all_rot_scales"]
        all_trans_scales = cache["all_trans_scales"]
        num_t = reverse_steps.shape[0]

        rigids_buffer = sample_feats['rigids_t'].clone().contiguous()
        sample_feats['rigids_t'] = rigids_buffer
        current_rigid_obj = ru.Rigid.from_tensor_7_fast(rigids_buffer)

        noise_shape = (len(reverse_steps),) + sample_feats['rigids_t'].shape[:-1] + (3,)
        if z_rot_all is None:
            z_rot_all = torch.randn(noise_shape, device=device)
        if z_trans_all is None:
            z_trans_all = torch.randn(noise_shape, device=device)

        all_rigids = []
        all_bb_prots = []
        t_start = time.perf_counter()
        with torch.no_grad():
            if self._model_conf.embed.embed_self_conditioning and self_condition:
                sample_feats['t'] = cache["sc_t"] * t_placeholder
                sample_feats['rot_score_scaling'] = cache["sc_rot_scale"] * t_placeholder
                sample_feats['trans_score_scaling'] = cache["sc_trans_scale"] * t_placeholder
                sample_feats = self._self_conditioning(sample_feats)
            for step_idx, t in enumerate(reverse_steps):
                rigids_buffer.copy_(current_rigid_obj.to_tensor_7().detach())
                if step_idx < len(reverse_steps) - 1:
                    sample_feats['t'] = t * t_placeholder
                    sample_feats['rot_score_scaling'] = all_rot_scales[step_idx] * t_placeholder
                    sample_feats['trans_score_scaling'] = all_trans_scales[step_idx] * t_placeholder
                    model_out = self.model(sample_feats)
                    rot_score = model_out['rot_score']
                    trans_score = model_out['trans_score']
                    rigid_pred = model_out['rigids']
                    if self._model_conf.embed.embed_self_conditioning:
                        sample_feats['sc_ca_t'] = rigid_pred[..., 4:]
                    fixed_mask = sample_feats['fixed_mask'] * sample_feats['res_mask']
                    diffuse_mask = (1 - sample_feats['fixed_mask']) * sample_feats['res_mask']
                    with autocast(enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                        current_rigid_obj = self.diffuser.reverse(
                            rigid_t=current_rigid_obj,
                            rot_score=rot_score,
                            trans_score=trans_score,
                            diffuse_mask=diffuse_mask,
                            t=t,
                            dt=dt,
                            sqrt_dt=sqrt_dt,
                            z_rot=z_rot_all[step_idx],
                            z_trans=z_trans_all[step_idx],
                            center=center,
                            noise_scale=noise_scale,
                            device=device,
                        )
                else:
                    model_out = self.model(sample_feats)
                    current_rigid_obj = ru.Rigid.from_tensor_7_fast(model_out['rigids'])
                if aux_traj:
                    all_rigids.append(model_out['rigids'])

                if step_idx == len(reverse_steps) - 1:
                    angles = model_out['angles']
                    atom37_t = all_atom.compute_backbone_atom37(
                        bb_rigids=current_rigid_obj,
                        aatypes=sample_feats['aatype'],
                        torsions=angles
                    )[0]
                    all_bb_prots.append(atom37_t)

        inference_time = time.perf_counter() - t_start
        print(f"inference_time:{inference_time:.2f} | num_t:{num_t} | noise_scale:{noise_scale}")

        def safe_flip(x):
            if len(x) > 0:
                return torch.flip(torch.stack(x), dims=(0,))
            return x

        all_bb_prots = safe_flip(all_bb_prots)
        all_rigids = safe_flip(all_rigids)
        ret = {
            'prot_traj': all_bb_prots,
            'rigid_traj': all_rigids
        }
        return ret


    def _calc_rot_trans_error(self, pred_rigids, gt_rigids, ref_rigids=None):
        """Calculate rotation/translation errors between prediction and ground truth.
        
        Args:
            pred_rigids: Predicted rigids (T, N, 7)
            gt_rigids: Ground truth rigids (T, N, 7)
            ref_rigids: Optional reference rigids (N, 7) for temporal comparison
        
        Returns:
            Tuple of (ave_rot, ave_trans, ref_ave_rot, ref_ave_trans, time_rot_dif, time_trans_dif)
        """
        # pred out
        average_quat_distances = average_quaternion_distances(gt_rigids[...,:4], pred_rigids[...,:4])
        average_trans_distances = average_translation_distances(gt_rigids[...,4:], pred_rigids[...,4:], measurement='MAE')
        
        # ref frame out (only if ref_rigids provided, else use zeros)
        if ref_rigids is not None:
            first_gt_rigids_expands = np.repeat(ref_rigids[np.newaxis, :, :], len(gt_rigids), axis=0)
            ref_average_quat_distances = average_quaternion_distances(gt_rigids[...,:4], first_gt_rigids_expands[...,:4])
            ref_average_trans_distances = average_translation_distances(gt_rigids[...,4:], first_gt_rigids_expands[...,4:], measurement='MAE')
        else:
            # No reference for single-frame prediction
            ref_average_quat_distances = 0.0
            ref_average_trans_distances = 0.0
        
        # caculate relative motion (temporal differences)
        if len(gt_rigids) > 1:
            time_rot_dif = average_quaternion_distances(gt_rigids[...,:4], np.roll(gt_rigids[...,:4], shift=1, axis=0))
            time_trans_dif = average_translation_distances(gt_rigids[...,4:], np.roll(gt_rigids[...,4:], shift=1, axis=0), measurement='MAE')
        else:
            # Single frame - no temporal differences
            time_rot_dif = 0.0
            time_trans_dif = 0.0

        return average_quat_distances, average_trans_distances, ref_average_quat_distances, ref_average_trans_distances, time_rot_dif, time_trans_dif

    def _prepare_eval_dirs(self, eval_dir, is_training):
        """Create and return all necessary evaluation directories."""
        dirs = {
            "sample": os.path.join(eval_dir, "sample"),
            "gt": os.path.join(eval_dir, "gt"),
            "pred_npz": os.path.join(eval_dir, "pred_npz") if not is_training else None,
        }
        for path in dirs.values():
            if path:
                os.makedirs(path, exist_ok=True)
        return dirs

    def _init_eval_metrics(self):
        """Initialize containers for evaluation metrics."""
        return {
            "metric_list": [],
            "metric_all_list": [],
            "metric_aligned_list": [],
            "metric_aligned_all_list": [],
            "first_frame_all_list": [],
            "save_name_list": [],
            "start_index_list": [],
            "rot_trans_error_dict": {
                "name": [], "ave_rot": [], "ave_trans": [],
                "first_rot": [], "first_trans": [],
                "time_rot_dif": [], "time_trans_dif": [],
            },
            "save_pdb_dict": {},
        }

    def _process_one_protein_for_eval(
        self,
        valid_feats,
        pdb_names,
        start_index,
        device,
        dirs,
        min_t=None,
        num_t=None,
        noise_scale=1.0,
        is_training=True,
    ):
        """Evaluate a single protein and compute metrics."""
        save_name = pdb_names[0].split(".")[0]
        frame_time = self._model_conf.frame_time
        sample_length = valid_feats["aatype"].shape[-1]
        diffuse_mask = np.ones(sample_length)
        b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

        # === Step 1. prepare init feats ===
        init_feats = self._prepare_init_feats(valid_feats, device, frame_time, sample_length)

        # === Step 2. inference ===
        sample_out = self.inference_fn(init_feats, num_t=num_t, min_t=min_t, aux_traj=True, noise_scale=noise_scale)

        # === Step 3. alignment ===
        align_sample, align_metric_list = self._align_predictions(sample_out["prot_traj"][0], valid_feats)

        # === Step 4. compute metrics ===
        result_metrics = self._compute_metrics(valid_feats, sample_out, align_sample, frame_time)

        # === Step 5. save predictions ===
        pdb_paths = self._save_eval_outputs(
            save_name, valid_feats, sample_out, align_sample, dirs, b_factors, is_training
        )

        # === Step 6. rotation / translation error (simplified for single-frame) ===
        # Note: ref_rigids_0 not used for seq->structure model
        rigid_traj = sample_out["rigid_traj"][0] if sample_out["rigid_traj"] else None
        if rigid_traj is not None and torch.is_tensor(rigid_traj):
            rigid_traj = rigid_traj.detach().cpu().numpy()
        
        # For single-frame structure prediction, use simple comparison
        if rigid_traj is not None:
            rot_trans_err = self._calc_rot_trans_error(
                rigid_traj,
                gt_rigids=init_feats["rigids_0"].cpu().numpy(),
                ref_rigids=None,
            )
        else:
            # Return empty metrics if no trajectory
            rot_trans_err = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return {
            "save_name": save_name,
            "start_index": start_index,
            "metrics": result_metrics,
            "pdb_paths": pdb_paths,
            "rot_trans_err": rot_trans_err,
        }

    def _accumulate_metrics(self, metrics, result):
        """Accumulate results for later aggregation."""
        metrics["save_name_list"].append(result["save_name"])
        metrics["start_index_list"].append(result["start_index"])
        metrics["metric_list"].append(result["metrics"]["mean_eval"])
        metrics["metric_all_list"].append(result["metrics"]["all_eval"])
        metrics["metric_aligned_list"].append(result["metrics"]["align_mean_eval"])
        metrics["metric_aligned_all_list"].append(result["metrics"]["align_all_eval"])
        metrics["first_frame_all_list"].append(result["metrics"]["first_frame_eval"])
        metrics["save_pdb_dict"][result["save_name"]] = result["pdb_paths"]

        ave_quat, ave_trans, ref_ave_quat, ref_ave_trans, time_rot_dif, time_trans_dif = result["rot_trans_err"]
        rdict = metrics["rot_trans_error_dict"]
        rdict["name"].append(result["save_name"])
        rdict["ave_rot"].append(ave_quat)
        rdict["ave_trans"].append(ave_trans)
        rdict["first_rot"].append(ref_ave_quat)
        rdict["first_trans"].append(ref_ave_trans)
        rdict["time_rot_dif"].append(time_rot_dif)
        rdict["time_trans_dif"].append(time_trans_dif)

    def _finalize_eval_outputs(self, metrics, dirs, eval_dir):
        """Aggregate metrics, plot results, and return evaluation summary."""
        # === change the DataFrame ===
        ckpt_eval_metrics = pd.DataFrame(metrics["metric_list"])
        ckpt_eval_metrics.insert(0, "pdb_name", metrics["save_name_list"])
        ckpt_eval_metrics.to_csv(os.path.join(eval_dir, "metrics.csv"), index=False)

        ckpt_eval_metrics_aligned = pd.DataFrame(metrics["metric_aligned_list"])
        ckpt_eval_metrics_aligned.insert(0, "pdb_name", metrics["save_name_list"])
        ckpt_eval_metrics_aligned.to_csv(os.path.join(eval_dir, "metrics_aligned.csv"), index=False)

        RefAsInfer_DF = pd.DataFrame(metrics["first_frame_all_list"])
        RefAsInfer_DF.insert(0, "pdb_name", metrics["save_name_list"])
        # === Comparison Curver ===
        metric_merge_dict = {
            "Pred": ckpt_eval_metrics,
            "RefAsInfer": RefAsInfer_DF,
        }

        curve_fig = plot_curve_merged(metric_merge_dict, eval_dir, row_num=3, col_num=len(metrics["save_name_list"]))
        curve_fig_aligned = plot_curve_merged(metric_merge_dict, eval_dir, row_num=3, col_num=len(metrics["save_name_list"]), suffer_fix="aligned")
        error_fig = plot_rot_trans_curve(metrics["rot_trans_error_dict"], save_path=eval_dir, frame_step=self._data_conf.frame_sample_step)

        # === Get the Average Score ===
        ckpt_eval_metrics = ckpt_eval_metrics.applymap(
            lambda x: float(x) if isinstance(x, np.ndarray) else x
        )
        mean_dict = ckpt_eval_metrics.drop(columns=["pdb_name"]).mean().to_dict()
        rot_trans_error_mean = {k: np.mean(v) for k, v in metrics["rot_trans_error_dict"].items() if k != "name"}

        model_ckpt_update = self._update_best_model(mean_dict, rot_trans_error_mean)

        self._log_eval_summary(mean_dict, rot_trans_error_mean)

        return {
            "metrics": ckpt_eval_metrics,
            "fig_dis_curve": curve_fig,
            "fig_dis_curve_aligned": curve_fig_aligned,
            "fig_error": error_fig,
            "model_ckpt_update": model_ckpt_update,
            "rot_trans_error_mean": rot_trans_error_mean,
            "save_pdb_dict": metrics["save_pdb_dict"],
        }

    def _align_predictions(self, pred_traj, valid_feats):
        """
        Align each frame of predicted trajectory to the reference CA positions.
        Args:
            pred_traj: np.ndarray or torch.Tensor, shape [T, N, 37, 3] or [N, 37, 3]
            valid_feats: dict containing 'atom37_mask'
        Returns:
            aligned_sample: torch.Tensor [T, N, 37, 3] or [N, 37, 3]
            align_metric_list: list of (rot, trans)
        """
        if torch.is_tensor(pred_traj):
            pred_traj = pred_traj.detach().cpu().numpy()

        # Handle single-frame case (seq->structure model)
        if "ref_atom37_pos" in valid_feats and valid_feats["ref_atom37_pos"] is not None:
            ref_ca = valid_feats["ref_atom37_pos"][0][0].cpu().numpy()[:, 1]  # reference CA
        elif "atom37_pos" in valid_feats:
            # Use ground truth as reference for alignment
            ref_ca = valid_feats["atom37_pos"][0][0].cpu().numpy()[:, 1] if valid_feats["atom37_pos"].dim() > 2 else valid_feats["atom37_pos"][0].cpu().numpy()[:, 1]
        else:
            # Use first frame of prediction as reference
            ref_ca = pred_traj[0][:, 1] if pred_traj.ndim == 4 else pred_traj[:, 1]
        
        atom37_mask = valid_feats["atom37_mask"][0].cpu().numpy() if valid_feats["atom37_mask"].dim() > 2 else valid_feats["atom37_mask"][0].cpu().numpy()

        align_sample_list, align_metric_list = [], []

        for frame_idx in range(pred_traj.shape[0]):
            sup = SVDSuperimposer()
            sup.set(ref_ca, pred_traj[frame_idx][:, 1])  # align on CA
            sup.run()
            rot, trans = sup.get_rotran()

            align_metric_list.append((rot, trans))
            aligned = np.dot(pred_traj[frame_idx], rot) + trans
            aligned *= atom37_mask[frame_idx][..., None]  # apply mask
            align_sample_list.append(torch.from_numpy(aligned))

        aligned_sample = torch.stack(align_sample_list)
        return aligned_sample, align_metric_list

    def _compute_metrics(self, valid_feats, sample_out, align_sample, frame_time):
        """
        Compute validation metrics for predicted, aligned, and reference trajectories.
        """
        gt_pos = valid_feats["atom37_pos"][0]
        gt_mask = valid_feats["atom37_mask"][0]
        ref_pos = valid_feats["ref_atom37_pos"][0, -1]  # last reference frame
        aatype = valid_feats["aatype"][0].cpu().numpy()

        # === Reference metrics (Ref as fake prediction)
        fake_res = torch.stack([ref_pos] * frame_time)
        first_eval_dic = compute_validation_metrics_all(
            gt_pos=gt_pos,
            out_pos=fake_res.cpu().numpy(),
            gt_mask=gt_mask,
            superimposition_metrics=True,
        )

        # === Unaligned prediction metrics
        pred_out = sample_out["prot_traj"][0]
        if torch.is_tensor(pred_out):
            pred_out = pred_out.detach().cpu().numpy()
        eval_dic = compute_validation_metrics_all(
            gt_pos=gt_pos,
            out_pos=pred_out,
            gt_mask=gt_mask,
            superimposition_metrics=True,
        )
        mean_eval_dic = {k: sum(v) / len(v) for k, v in eval_dic.items()}

        # === Aligned prediction metrics
        align_eval_dic = compute_validation_metrics_all(
            gt_pos=gt_pos,
            out_pos=align_sample.cpu().numpy(),
            gt_mask=gt_mask,
            superimposition_metrics=True,
        )
        align_mean_eval_dic = {k: sum(v) / len(v) for k, v in align_eval_dic.items()}

        return {
            "first_frame_eval": {
                k: (v.cpu().numpy() if torch.is_tensor(v) else v)
                for k, v in first_eval_dic.items()
            },
            "mean_eval": {
                k: (v.cpu().numpy() if torch.is_tensor(v) else v)
                for k, v in mean_eval_dic.items()
            },
            "all_eval": {
                k: (v.cpu().numpy() if torch.is_tensor(v) else v)
                for k, v in eval_dic.items()
            },
            "align_mean_eval": {
                k: (v.cpu().numpy() if torch.is_tensor(v) else v)
                for k, v in align_mean_eval_dic.items()
            },
            "align_all_eval": {
                k: (v.cpu().numpy() if torch.is_tensor(v) else v)
                for k, v in align_eval_dic.items()
            },
        }

    def _save_eval_outputs(
        self,
        save_name,
        valid_feats,
        sample_out,
        align_sample,
        dirs,
        b_factors,
        is_training=True,
    ):
        """
        Save GT, predicted, and aligned structures in PDB and NPZ format.
        """
        gt_path = os.path.join(dirs["gt"], f"{save_name}_gt.pdb")
        sample_path = os.path.join(dirs["sample"], f"{save_name}.pdb")
        sample_aligned_path = os.path.join(dirs["sample"], f"{save_name}_aligned.pdb")
        first_motion_path = os.path.join(dirs["sample"], f"{save_name}_first_motion.pdb")

        def _to_np_1d(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                x = x.detach().cpu()
                if x.dim() > 1:
                    x = x[0]
                return x.numpy()
            x = np.asarray(x)
            if x.ndim > 1:
                x = x[0]
            return x

        chain_index = _to_np_1d(valid_feats.get("pdb_chain_index", None))
        residue_index = _to_np_1d(valid_feats.get("pdb_residue_index", None))

        # === save GT
        aatype = valid_feats["aatype"][0, 0].cpu().numpy()
        au.write_prot_to_pdb(
            prot_pos=valid_feats["atom37_pos"][0].cpu().numpy(),
            file_path=gt_path,
            aatype=aatype,
            no_indexing=True,
            b_factors=b_factors,
            residue_index=residue_index,
            chain_index=chain_index,
        )

        # === save aligned prediction
        au.write_prot_to_pdb(
            prot_pos=align_sample.cpu().numpy(),
            file_path=sample_aligned_path,
            aatype=aatype,
            no_indexing=True,
            b_factors=b_factors,
            residue_index=residue_index,
            chain_index=chain_index,
        )

        # === save unaligned prediction
        pred_out = sample_out["prot_traj"][0]
        if torch.is_tensor(pred_out):
            pred_out = pred_out.detach().cpu().numpy()
        au.write_prot_to_pdb(
            prot_pos=pred_out,
            file_path=sample_path,
            aatype=aatype,
            no_indexing=True,
            b_factors=b_factors,
            residue_index=residue_index,
            chain_index=chain_index,
        )

        # === save first_motion (optional, only if motion/ref data exists)
        if "motion_atom37_pos" in valid_feats and "ref_atom37_pos" in valid_feats:
            au.write_prot_to_pdb(
                prot_pos=np.concatenate(
                    (
                        valid_feats["motion_atom37_pos"][0].cpu().numpy(),
                        valid_feats["ref_atom37_pos"][0].cpu().numpy(),
                    ),
                    axis=0,
                ),
                file_path=first_motion_path,
                aatype=aatype,
                no_indexing=True,
                b_factors=b_factors,
                residue_index=residue_index,
                chain_index=chain_index,
            )

        # === save npz if not training
        if not is_training and dirs.get("pred_npz"):

            # also save "first" reference frame for visualization (if available)
            if "ref_atom37_pos" in valid_feats:
                au.write_prot_to_pdb(
                    prot_pos=valid_feats["ref_atom37_pos"][0].cpu().numpy(),
                    file_path=os.path.join(dirs["sample"], f"{save_name}_first.pdb"),
                    aatype=aatype,
                    no_indexing=True,
                    b_factors=b_factors,
                    residue_index=residue_index,
                    chain_index=chain_index,
                )

        return {"gt": gt_path, "gen": sample_path}

    def _prepare_extension_eval_dirs(self, eval_dir):
        pdb_base_path = os.path.join(eval_dir, "extension_pdb")
        ref_base_path = os.path.join(eval_dir, "reference_pdb")
        os.makedirs(pdb_base_path, exist_ok=True)
        os.makedirs(ref_base_path, exist_ok=True)
        return pdb_base_path, ref_base_path

    def _process_one_protein_extrapolation(
        self,
        extrapolation_time,
        valid_feats,
        pdb_names,
        ref_base_path,
        pdb_base_path,
        device,
        min_t=None,
        num_t=None,
        noise_scale=1.0,
        executor=None,
    ):
        """Process one protein sequence and perform trajectory extrapolation.
        
        Note: This method is disabled for seq->structure models (frame_time=1, ref_number=0, motion_number=0)
        """
        # Check if extrapolation is supported (requires temporal data)
        frame_time = self._model_conf.frame_time
        ref_number = self._model_conf.ref_number
        motion_number = self._model_conf.motion_number
        
        if frame_time == 1 and ref_number == 0 and motion_number == 0:
            self._log.warning(f"Skipping extrapolation for {pdb_names[0]}: not supported for seq->structure model")
            return

        # === Preparation ===
        protein_name = pdb_names[0]

        aatype = valid_feats["aatype"].cpu().numpy()
        sample_length = aatype.shape[-1]
        b_factors = np.tile((np.ones(sample_length) * 100)[:, None], (1, 37))
        pdb_path = os.path.join(pdb_base_path, f"{protein_name}_time_{extrapolation_time}.pdb")
        if os.path.exists(pdb_path):
            print(f"✅ {protein_name} already existed in: {pdb_path}")
            return
        
        # Save reference structure (if available)
        if "ref_atom37_pos" not in valid_feats:
            self._log.warning(f"Skipping extrapolation for {pdb_names[0]}: ref_atom37_pos not available")
            return
        ref_all_atom_positions = valid_feats["ref_atom37_pos"][0].cpu().numpy()

        def _to_np_1d(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                x = x.detach().cpu()
                # valid_feats 通常是 (B,...)；如果是 (B,T,...) 这里取第一个 batch
                if x.dim() > 1:
                    x = x[0]
                return x.numpy()
            x = np.asarray(x)
            if x.ndim > 1:
                x = x[0]
            return x

        chain_index = _to_np_1d(valid_feats.get("pdb_chain_index", None))
        residue_index = _to_np_1d(valid_feats.get("pdb_residue_index", None))

        au.write_prot_to_pdb(
            prot_pos=ref_all_atom_positions,
            file_path=os.path.join(ref_base_path, f"{protein_name}.pdb"),
            aatype=aatype[0, 0],
            no_indexing=True,
            b_factors=b_factors,
            residue_index=residue_index,
            chain_index=chain_index,
        )

        print(f"[Eval] Processing {protein_name}, length={extrapolation_time}")

        # === Initialize input ===
        atom_traj, rigid_traj = [], []
        valid_feats = self._prepare_init_feats(valid_feats, device, frame_time, sample_length)
        cache = self._get_inference_cache(device, num_t=num_t, min_t=min_t)
        reverse_steps = cache["reverse_steps"]
        all_start_rigids = self.diffuser.sample_ref(
            n_samples=extrapolation_time * frame_time * sample_length,
            as_tensor_7=True,
        )["rigids_t"].reshape(extrapolation_time, frame_time, sample_length, 7).to(device)
        z_rot_all = torch.randn(
            len(reverse_steps), frame_time, sample_length, 3, device=device
        )
        z_trans_all = torch.randn(
            len(reverse_steps), frame_time, sample_length, 3, device=device
        )

        # === Iterative inference ===
        pbar = tqdm(range(extrapolation_time), desc=f"{protein_name}", ncols=80)
        ref_update_mode = getattr(self._conf.eval, "ref_update_mode", "pre_motion")

        for j in pbar:
            # === Perform inference ===
            sample_out = self.inference_fn(
                valid_feats,
                num_t=num_t,
                min_t=min_t,
                aux_traj=True,
                noise_scale=noise_scale,
                z_rot_all=z_rot_all,
                z_trans_all=z_trans_all,
            )

            atom_pred = sample_out["prot_traj"][0]
            rigid_pred = sample_out["rigid_traj"][0]

            # Save the results
            atom_traj.append(atom_pred[-frame_time:])
            rigid_traj.append(rigid_pred[-frame_time:])

            # === Update reference state ===
            valid_feats["rigids_t"] = all_start_rigids[j]
            valid_feats = self._update_ref_with_prediction(
                valid_feats,
                atom_pred,
                rigid_pred,
                ref_number,
                motion_number,
                device,
                ref_update_mode=ref_update_mode,
            )

        # === Concatenate trajectory and save ===
        if torch.is_tensor(atom_traj[0]):
            atom_traj = torch.cat(atom_traj, dim=0).detach().cpu().numpy()
        else:
            atom_traj = np.concatenate(atom_traj, axis=0)
        if torch.is_tensor(rigid_traj[0]):
            rigid_traj = torch.cat(rigid_traj, dim=0).detach().cpu().numpy()
        else:
            rigid_traj = np.concatenate(rigid_traj, axis=0)

        if executor is not None:
            executor.submit(
                au.write_prot_to_pdb,
                prot_pos=atom_traj,
                file_path=pdb_path,
                aatype=aatype[0, 0],
                no_indexing=True,
                b_factors=b_factors,
                residue_index=residue_index,
                chain_index=chain_index,
            )
        else:
            au.write_prot_to_pdb(
                prot_pos=atom_traj,
                file_path=pdb_path,
                aatype=aatype[0, 0],
                no_indexing=True,
                b_factors=b_factors,
                residue_index=residue_index,
                chain_index=chain_index,
            )

    def _prepare_init_feats(self, valid_feats, device, frame_time, sample_length):
        """Prepare initial features for inference (multimer-safe)."""
        # masks
        res_mask = np.ones((frame_time, sample_length), dtype=np.float32)
        fixed_mask = np.zeros_like(res_mask, dtype=np.float32)

        # Prefer multimer-aware indices if present in valid_feats
        # valid_feats may already include:
        #   - seq_idx or residue_index_in_chain : (frame_time, N) or (1, frame_time, N)
        #   - asym_id/entity_id/sym_id          : (frame_time, N) or (1, frame_time, N)
        # If not present, fallback to 1..N (single chain).
        if "residue_index_in_chain" in valid_feats:
            seq_idx = valid_feats["residue_index_in_chain"]
            if torch.is_tensor(seq_idx):
                # expect shape (1, frame_time, N) or (frame_time, N)
                if seq_idx.dim() == 3:
                    seq_idx = seq_idx[0]
                seq_idx = seq_idx.to(device)
            else:
                seq_idx = torch.tensor(seq_idx, device=device)
        elif "seq_idx" in valid_feats:
            seq_idx = valid_feats["seq_idx"]
            if torch.is_tensor(seq_idx):
                if seq_idx.dim() == 3:
                    seq_idx = seq_idx[0]
                seq_idx = seq_idx.to(device)
            else:
                seq_idx = torch.tensor(seq_idx, device=device)
        else:
            seq_idx = torch.arange(1, sample_length + 1, device=device).unsqueeze(0).repeat(frame_time, 1)

        # sample reference rigids_t
        ref_sample = self.diffuser.sample_ref(
            n_samples=sample_length * frame_time,
            as_tensor_7=True,
        )
        ref_sample["rigids_t"] = ref_sample["rigids_t"].reshape([-1, frame_time, sample_length, 7])
        ref_sample = tree.map_structure(lambda x: x.to(device), ref_sample)

        # pack input
        init_feats = {
            "res_mask": torch.tensor(res_mask[None], device=device),
            "seq_idx": seq_idx[None],  # (1, T, N)
            "fixed_mask": torch.tensor(fixed_mask[None], device=device),
            "torsion_angles_sin_cos": torch.zeros((1, sample_length, 7, 2), device=device),
            "sc_ca_t": torch.zeros((1, frame_time, sample_length, 3), device=device),
            **{k: (v.to(device) if torch.is_tensor(v) else torch.tensor(v, device=device))
               for k, v in valid_feats.items()},
            **ref_sample,
        }

        # Safer packing for validation/inference:
        T = None
        if "res_mask" in init_feats and torch.is_tensor(init_feats["res_mask"]) and init_feats["res_mask"].dim() >= 3:
            # res_mask is expected to be (B, T, N)
            T = init_feats["res_mask"].shape[1]

        for key, value in list(init_feats.items()):
            if not torch.is_tensor(value):
                continue
            if key == "t":
                continue

            if T is not None and value.dim() >= 3 and value.shape[1] == T:
                init_feats[key] = value.flatten(0, 1)     # (B*T, ...)
            else:
                if value.dim() >= 2 and value.shape[0] == 1:
                    init_feats[key] = value.squeeze(0)

        return init_feats

    def _update_ref_with_prediction(
        self,
        valid_feats,
        atom_pred,
        rigid_pred,
        ref_number,
        motion_number,
        device,
        change_ref=False,
        ref_update_mode=None,
    ):
        """Update reference features with newly predicted frames.
        
        Note: For seq->structure model (ref_number=0, motion_number=0), this is a no-op.
        """
        # Skip if ref_number and motion_number are both 0 (seq->structure model)
        if ref_number == 0 and motion_number == 0:
            return valid_feats
        
        rigid_pred_t = rigid_pred if torch.is_tensor(rigid_pred) else torch.as_tensor(rigid_pred, device=device)
        atom_pred_t = atom_pred if torch.is_tensor(atom_pred) else torch.as_tensor(atom_pred, device=device)
        
        # Handle missing motion/ref features (seq->structure model case)
        motion_rigids = valid_feats.get("motion_rigids_0", torch.empty(0, device=device))
        ref_rigids = valid_feats.get("ref_rigids_0", torch.empty(0, device=device))
        motion_atoms = valid_feats.get("motion_atom37_pos", torch.empty(0, device=device))
        ref_atoms = valid_feats.get("ref_atom37_pos", torch.empty(0, device=device))
        
        concat_rigids = torch.cat([
            motion_rigids.to(device) if motion_rigids.numel() > 0 else torch.empty(0, device=device),
            ref_rigids.to(device) if ref_rigids.numel() > 0 else torch.empty(0, device=device),
            rigid_pred_t.to(device).to(motion_rigids.dtype if motion_rigids.numel() > 0 else rigid_pred_t.dtype),
        ], dim=0)

        concat_atoms = torch.cat([
            motion_atoms.to(device) if motion_atoms.numel() > 0 else torch.empty(0, device=device),
            ref_atoms.to(device) if ref_atoms.numel() > 0 else torch.empty(0, device=device),
            atom_pred_t.to(device).to(motion_atoms.dtype if motion_atoms.numel() > 0 else atom_pred_t.dtype),
        ], dim=0)
        if ref_update_mode is None:
            ref_update_mode = "best_pred" if change_ref else "keep"

        if ref_update_mode == "pre_motion":
            if rigid_pred_t.shape[0] < (ref_number + motion_number):
                raise ValueError(
                    "pre_motion reference update requires the prediction block to contain at least "
                    "ref_number + motion_number frames."
                )
            ref_start = -(motion_number + ref_number)
            ref_end = -motion_number if motion_number > 0 else None
            valid_feats["ref_rigids_0"] = rigid_pred_t[ref_start:ref_end].to(valid_feats["ref_rigids_0"].dtype)
            valid_feats["ref_atom37_pos"] = atom_pred_t[ref_start:ref_end].to(valid_feats["ref_atom37_pos"].dtype)
        elif ref_update_mode == "first_pred":
            valid_feats["ref_rigids_0"] = rigid_pred_t[:ref_number].to(valid_feats["ref_rigids_0"].dtype)
            valid_feats["ref_atom37_pos"] = atom_pred_t[:ref_number].to(valid_feats["ref_atom37_pos"].dtype)
        elif ref_update_mode == "keep":
            pass
        elif ref_update_mode == "best_pred":
            # valid_feats["ref_rigids_0"] = concat_rigids[-ref_number:]#.unsqueeze(0)
            # valid_feats["ref_atom37_pos"] = concat_atoms[-ref_number:]#.unsqueeze(0)
            lddt_score = lddt_ca(
                all_atom_positions=valid_feats["ref_atom37_pos"],
                all_atom_pred_pos=concat_atoms[ref_number + motion_number :],
                all_atom_mask=valid_feats['atom37_mask'],
                per_residue=False
            )
            def compute_ca_clash_score_batch(atom_positions, atom_mask, clash_threshold=2.0):
                CA_IDX = residue_constants.atom_order["CA"]
                ca_pos = atom_positions[:, :, CA_IDX, :]
                ca_mask = atom_mask[:, :, CA_IDX].bool()
                diff = ca_pos.unsqueeze(2) - ca_pos.unsqueeze(1)
                dist = torch.linalg.norm(diff, dim=-1)
                T, N = ca_mask.shape
                dist = dist + torch.eye(N, device=dist.device).unsqueeze(0) * 999.0
                valid_pairs = ca_mask.unsqueeze(2) & ca_mask.unsqueeze(1)
                clashes = (dist < clash_threshold) & valid_pairs
                clash_atoms = clashes.any(dim=-1)
                clash_ratio = clash_atoms.sum(dim=-1) / ca_mask.sum(dim=-1)
                return clash_ratio
            clash_ratio = compute_ca_clash_score_batch(
                concat_atoms[ref_number + motion_number :],
                valid_feats["atom37_mask"],
                clash_threshold=2.0,
            )
            idx1 = torch.argsort(clash_ratio)
            sorted_idx = idx1[torch.argsort(lddt_score[idx1])]
            best_idx = sorted_idx[0].item()
            if clash_ratio[best_idx] < 0.01:
                valid_feats["ref_rigids_0"] = concat_rigids[ref_number + motion_number :][
                    best_idx : best_idx + 1
                ]#.unsqueeze(0)
                valid_feats["ref_atom37_pos"] = concat_atoms[ref_number + motion_number :][
                    best_idx : best_idx + 1
                ]#.unsqueeze(0)
                # select which ref to be
                self._log.info(
                    f"Changed reference to candidate {best_idx} with generated segment shape "
                    f"{tuple(concat_atoms[ref_number + motion_number :].shape)}."
                )
            else:
                self._log.info("Generated segment has clash; keeping the original reference.")
        else:
            raise ValueError(
                f"Unsupported ref_update_mode={ref_update_mode!r}. "
                "Expected one of {'keep', 'pre_motion', 'first_pred', 'best_pred'}."
            )

        # Update motion features only if motion_number > 0
        if motion_number > 0 and "motion_rigids_0" in valid_feats:
            valid_feats["motion_rigids_0"] = rigid_pred_t[-motion_number:].to(valid_feats["motion_rigids_0"].dtype)
            valid_feats["motion_atom37_pos"] = atom_pred_t[-motion_number:].to(valid_feats["motion_atom37_pos"].dtype)

        return valid_feats

    def _update_best_model(self, mean_dict: dict, rot_trans_error_mean: dict) -> bool:
        """
        Compare current evaluation metrics with previous best; update if improved.
        """
        model_ckpt_update = False

        # Determine if improved (thresholds can be adjusted as needed)
        better_rmse = mean_dict["rmse_all"] < self.best_rmse_all
        better_rot = rot_trans_error_mean["ave_rot"] < self.best_rot_error
        better_trans = rot_trans_error_mean["ave_trans"] < self.best_trans_error

        if better_rmse or better_rot or better_trans:
            self.best_rmse_all = mean_dict["rmse_all"]
            self.best_rmse_ca = mean_dict["rmse_ca"]
            self.best_drmsd = mean_dict["drmsd_ca"]
            self.best_rmsd_ca_aligned = mean_dict["rmsd_ca_aligned"]

            self.best_rot_error = rot_trans_error_mean["ave_rot"]
            self.best_trans_error = rot_trans_error_mean["ave_trans"]
            self.best_ref_rot_error = rot_trans_error_mean["first_rot"]
            self.best_ref_trans_error = rot_trans_error_mean["first_trans"]

            self.best_trained_steps = self.trained_steps
            self.best_trained_epoch = self.trained_epochs
            model_ckpt_update = True

        return model_ckpt_update

    def _log_eval_summary(self, mean_dict: dict, rot_trans_error_mean: dict) -> None:
        """
        Print evaluation summary and current best metrics.
        """
        info = f"Step:{self.trained_steps} "
        for k, v in mean_dict.items():
            info += f"avg_{k}:{v:.4f} "
        for k, v in rot_trans_error_mean.items():
            if k != "name":
                info += f"avg_{k}:{v:.4f} "

        self._log.info("Evaluation Results: " + info)
        self._log.info(
            f"Best so far | steps/epoch: {self.best_trained_steps}/{self.best_trained_epoch} | "
            f"rmse_all: {self.best_rmse_all:.4f} | "
            f"rmse_ca: {self.best_rmse_ca:.4f} | "
            f"rmsd_ca_aligned: {self.best_rmsd_ca_aligned:.4f} | "
            f"drmsd_ca: {self.best_drmsd:.4f} | "
            f"rot_error: {self.best_rot_error:.4f}/{self.best_ref_rot_error:.4f} | "
            f"trans_error: {self.best_trans_error:.4f}/{self.best_ref_trans_error:.4f}"
        )

@hydra.main(version_base=None, config_path="./config", config_name="train_DyneTrion")
def run(conf: DictConfig) -> None:

    exp = Experiment(conf=conf)
    exp.start_training()


if __name__ == '__main__':
    run()
