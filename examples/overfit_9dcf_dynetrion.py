#!/usr/bin/env python3
"""Overfit DyneTrion on 9DCF protein-RNA complex."""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
os.environ['PYTHONUNBUFFERED'] = '1'

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from torch.utils import data
import wandb
from openfold.utils import rigid_utils as ru

# Set random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Monkey-patch swanlab -> wandb
class _SwanlabShim:
    @staticmethod
    def login(*args, **kwargs):
        pass
    @staticmethod
    def init(project=None, experiment_name=None, config=None, mode=None, logdir=None, **kwargs):
        return wandb.init(project=project, name=experiment_name, config=config, settings=wandb.Settings(console="off"))
    @staticmethod
    def log(*args, **kwargs):
        return wandb.log(*args, **kwargs)
    class Image:
        def __init__(self, data, *args, **kwargs):
            self._data = data
        def __getattr__(self, name):
            return getattr(wandb.Image(self._data), name)

sys.modules['swanlab'] = _SwanlabShim()

import DyneTrion.train_DyneTrion as train_DyneTrion
from src.data import DyneTrion_data_loader_dynamic


def kabsch_align(P, Q):
    """Kabsch alignment of P onto Q."""
    P = P - P.mean(axis=0, keepdims=True)
    Q = Q - Q.mean(axis=0, keepdims=True)
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    aligned = P @ R
    return aligned, R


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    csv_path = 'overfitting_data/9DCF/9dcf_data.csv'
    
    base_conf = OmegaConf.load('DyneTrion/config/train_DyneTrion.yaml')
    
    overrides = {
        'experiment': {
            'name': 'overfit_9dcf',
            'learning_rate': 1e-4,
            'num_epoch': 1,
            'batch_size': 1,
            'eval_batch_size': 1,
            'num_loader_workers': 1,
            'log_freq': 50,
            'ckpt_dir': 'outputs/overfit_9dcf/ckpt',
            'eval_dir': 'outputs/overfit_9dcf/eval',
            'enable_validation': False,
            'training': True,
            'use_recoder': True,
            'use_ddp': False,
            'use_gpu': True,
            'num_gpus': 1,
            'device': device,
            'warm_start': None,
            'smooth_lddt_loss_weight': 0.0,
            'trans_loss_weight': 1.0,
            'rot_loss_weight': 1.0,
            'torsion_loss_weight': 0.0,
            'bb_atom_loss_weight': 0.0,
            'dist_mat_loss_weight': 0.0,
            'clip_grad': True,
            'clip_grad_norm': 1.0,
            'coordinate_scaling': 0.1,
        },
        'data': {
            **base_conf.data,
            'csv_path': csv_path,
            'val_csv_path': csv_path,
            'test_csv_path': csv_path,
            'max_protein_num': 1000,
            'crop': {'enabled': False},
            'frame_time': 1,
            'ref_number': 0,
            'motion_number': 0,
            'filtering': {
                'train_max_len': 2048,
                'val_max_len': 2048,
                'test_max_len': 2048,
            },
        },
    }
    
    conf = OmegaConf.merge(base_conf, OmegaConf.create(overrides))
    OmegaConf.set_struct(conf, False)
    
    print("\nCreating DyneTrion Experiment...")
    exp = train_DyneTrion.Experiment(conf=conf)
    exp._available_gpus = '0'
    
    # Monkey-patch swanlab logger
    def init_wandb_logger():
        conf_dict = OmegaConf.to_container(exp._conf, resolve=False)
        wandb.init(
            project="dynetrion-protein-rna",
            name="overfit_9dcf",
            config=conf_dict,
            settings=wandb.Settings(console="off")
        )
        exp.swanlab_logger = wandb
    exp.init_swanlab_logger = init_wandb_logger
    
    # Monkey-patch create_dataset to avoid prefetch_factor issue with num_workers
    def patched_create_dataset():
        train_dataset = DyneTrion_data_loader_dynamic.PdbDataset(
            data_conf=exp._data_conf,
            diffuser=exp._diffuser,
            is_training=True,
        )
        valid_dataset = DyneTrion_data_loader_dynamic.PdbDataset(
            data_conf=exp._data_conf,
            diffuser=exp._diffuser,
            is_training=False,
        )
        
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        valid_loader = data.DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        return train_loader, valid_loader
    
    exp.create_dataset = patched_create_dataset
    
    print("\nStarting training setup...")
    exp.start_training(return_logs=False)
    
    actual_device = next(exp.model.parameters()).device
    print(f"Model is on: {actual_device}")
    
    train_loader, valid_loader = exp.create_dataset()
    
    # Overfit for 500 epochs
    num_epochs = 200
    print(f"\nOverfitting for {num_epochs} epochs on 9DCF...")
    print(f"Learning rate: {exp._optimizer.param_groups[0]['lr']}")
    
    for epoch in range(num_epochs):
        exp.train_epoch(train_loader, valid_loader, actual_device, return_logs=False)
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}/{num_epochs}")
    
    print(f"\nTraining complete. Total steps: {exp.trained_steps}")
    
    # Save checkpoint
    os.makedirs('outputs/overfit_9dcf', exist_ok=True)
    ckpt_path = 'outputs/overfit_9dcf/final_model.pth'
    from src.data import utils as du
    du.write_checkpoint(
        ckpt_path,
        exp.model.state_dict(),
        conf,
        exp._optimizer.state_dict(),
        exp.trained_epochs,
        exp.trained_steps,
        use_torch=True
    )
    print(f"Saved checkpoint: {ckpt_path}")
    
    # Inference
    print("\nRunning inference...")
    exp.model.eval()
    
    infer_dataset = DyneTrion_data_loader_dynamic.PdbDataset(
        data_conf=exp._data_conf,
        diffuser=exp._diffuser,
        is_training=False,
    )
    infer_dataset.csv = pd.read_csv(csv_path).iloc[:1].reset_index(drop=True)
    
    eval_loader = data.DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=0)
    eval_batch = next(iter(eval_loader))
    
    # Handle variable return length from dataset
    if len(eval_batch) == 3:
        valid_feats, pdb_name, start_index = eval_batch
    else:
        valid_feats, pdb_name = eval_batch
        start_index = 0
    
    pdb_name = pdb_name[0] if isinstance(pdb_name, (list, tuple)) else str(pdb_name)
    print(f"Inference on: {pdb_name}")
    
    for k, v in list(valid_feats.items()):
        if torch.is_tensor(v):
            valid_feats[k] = v.to(actual_device)
    
    frame_time = exp._model_conf.frame_time
    sample_length = valid_feats["aatype"].shape[-1]
    
    init_feats = exp._prepare_init_feats(valid_feats, actual_device, frame_time, sample_length)
    
    if 'ref_rigids_0' in init_feats and init_feats['ref_rigids_0'].dim() == 2:
        init_feats['ref_rigids_0'] = init_feats['ref_rigids_0'].unsqueeze(0)
    
    sample_out = exp.inference_fn(
        init_feats,
        num_t=50,
        min_t=0.01,
        aux_traj=True,
        noise_scale=1.0,
    )
    
    # Get predictions and GT
    pred_rigids = ru.Rigid.from_tensor_7(sample_out["rigid_traj"][0][-1])
    pred_ca = pred_rigids.get_trans().squeeze(0).cpu().numpy()
    
    gt_rigids = ru.Rigid.from_tensor_7(valid_feats['rigids_0'].squeeze())
    gt_ca = gt_rigids.get_trans().cpu().numpy()
    
    # Kabsch alignment
    pred_ca_aligned, R = kabsch_align(pred_ca, gt_ca)
    ca_rmsd = np.sqrt(np.mean((pred_ca_aligned - gt_ca) ** 2))
    rmsd_raw = np.sqrt(np.mean((pred_ca - gt_ca) ** 2))
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  CA RMSD (raw): {rmsd_raw:.2f} A")
    print(f"  CA RMSD (Kabsch): {ca_rmsd:.2f} A")
    print(f"  Total steps: {exp.trained_steps}")
    print(f"{'='*60}")
    
    wandb.log({'result/ca_rmsd': ca_rmsd, 'result/ca_rmsd_raw': rmsd_raw})
    wandb.finish()


if __name__ == '__main__':
    main()
