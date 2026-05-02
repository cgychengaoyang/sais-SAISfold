#!/usr/bin/env python3
"""Run inference on 9DCF using the saved overfit checkpoint."""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
os.environ['PYTHONUNBUFFERED'] = '1'

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import re
from omegaconf import OmegaConf
from torch.utils import data
import wandb
from openfold.utils import rigid_utils as ru
from Bio.PDB import Structure, Model, Chain, Residue, Atom, PDBIO

# Monkey-patch swanlab -> wandb
class _SwanlabShim:
    @staticmethod
    def login(*args, **kwargs):
        pass
    @staticmethod
    def init(*args, **kwargs):
        return wandb.init(*args, **kwargs)
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

try:
    from rhofold.utils.constants import RNA_CONSTANTS
    HAS_RHOFOLD = True
except ImportError:
    HAS_RHOFOLD = False


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


def get_residue_name_3char(aatype_idx, is_rna):
    """Map token index to PDB residue name."""
    protein_names = {
        0: 'ALA', 1: 'ARG', 2: 'ASN', 3: 'ASP', 4: 'CYS',
        5: 'GLN', 6: 'GLU', 7: 'GLY', 8: 'HIS', 9: 'ILE',
        10: 'LEU', 11: 'LYS', 12: 'MET', 13: 'PHE', 14: 'PRO',
        15: 'SER', 16: 'THR', 17: 'TRP', 18: 'TYR', 19: 'VAL',
        20: 'UNK',
    }
    rna_names = {21: '  A', 22: '  G', 23: '  C', 24: '  U', 25: '  N'}
    if is_rna:
        return rna_names.get(int(aatype_idx), '  N')[:3]
    return protein_names.get(int(aatype_idx), 'UNK')


def get_element(atom_name):
    m = re.match(r'([A-Za-z]+)', str(atom_name))
    if not m:
        return 'C'
    elem = m.group(1).upper()
    if elem in ('CL', 'FE', 'BR', 'ZN', 'CA', 'MG', 'NA', 'CU', 'MN'):
        return elem
    return elem[0]


def build_full_atom_pdb(all_atom_pos, all_atom_mask, aatype, is_rna, chain_index,
                        residue_index, output_path, pdb_id='PRED'):
    """Build full-atom PDB from atom arrays using Bio.PDB."""
    N = all_atom_pos.shape[0]
    structure = Structure.Structure(pdb_id)
    model_obj = Model.Model(0)
    structure.add(model_obj)

    unique_chains = np.unique(chain_index)
    chains = {}
    for cid in unique_chains:
        chains[cid] = Chain.Chain(chr(ord('A') + int(cid)))
        model_obj.add(chains[cid])

    atom_serial = 1
    for i in range(N):
        if all_atom_mask[i].sum() == 0:
            continue
        cid = int(chain_index[i])
        res_num = int(residue_index[i]) if residue_index is not None else i + 1
        res_name = get_residue_name_3char(aatype[i], is_rna[i])
        residue = Residue.Residue((' ', res_num, ' '), res_name, 0)
        chains[cid].add(residue)

        if is_rna[i] and HAS_RHOFOLD:
            base = {21: 'A', 22: 'G', 23: 'C', 24: 'U', 25: 'N'}.get(int(aatype[i]), 'N')
            atom_names = RNA_CONSTANTS.ATOM_NAMES_PER_RESD.get(base, [])
            for j, atom_name in enumerate(atom_names):
                if j >= 37:
                    break
                if all_atom_mask[i, j] > 0:
                    coord = all_atom_pos[i, j]
                    atom = Atom.Atom(
                        atom_name, coord, 0.0, 1.0, ' ',
                        atom_name, atom_serial, get_element(atom_name)
                    )
                    residue.add(atom)
                    atom_serial += 1
        else:
            # Protein uses atom37 naming
            from openfold.np import residue_constants as rc
            for j in range(37):
                if all_atom_mask[i, j] > 0:
                    atom_name = rc.atom_types[j]
                    coord = all_atom_pos[i, j]
                    atom = Atom.Atom(
                        atom_name, coord, 0.0, 1.0, ' ',
                        atom_name, atom_serial, get_element(atom_name)
                    )
                    residue.add(atom)
                    atom_serial += 1

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path))
    print(f"Saved PDB: {output_path}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    csv_path = 'overfitting_data/9DCF/9dcf_data.csv'
    ckpt_path = 'outputs/overfit_9dcf/final_model.pth'
    out_dir = 'outputs/overfit_9dcf'
    os.makedirs(out_dir, exist_ok=True)

    base_conf = OmegaConf.load('DyneTrion/config/train_DyneTrion.yaml')

    overrides = {
        'experiment': {
            'name': 'overfit_9dcf',
            'learning_rate': 1e-4,
            'num_epoch': 1,
            'batch_size': 1,
            'eval_batch_size': 1,
            'num_loader_workers': 0,
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
            'warm_start': ckpt_path,
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

    def init_wandb_logger():
        wandb.init(project="dynetrion-protein-rna", name="inference_9dcf", settings=wandb.Settings(console="off"))
        exp.swanlab_logger = wandb
    exp.init_swanlab_logger = init_wandb_logger

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
        train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
        valid_loader = data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        return train_loader, valid_loader

    exp.create_dataset = patched_create_dataset

    print("\nStarting training setup (to load checkpoint)...")
    exp.start_training(return_logs=False)

    actual_device = next(exp.model.parameters()).device
    print(f"Model is on: {actual_device}")

    # Load ground truth structure for reconstruction
    struct_data = dict(np.load('overfitting_data/9DCF/structure.npz'))
    gt_atom37 = struct_data['all_atom_positions']
    gt_atom37_mask = struct_data['all_atom_mask']
    aatype_arr = struct_data['aatype']
    is_rna_arr = struct_data['is_rna']
    chain_index_arr = struct_data['chain_index']
    residue_index_arr = struct_data['residue_index']

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
    gt_rigids = ru.Rigid.from_tensor_7(valid_feats['rigids_0'].squeeze())

    aatype = valid_feats['aatype'].squeeze().long()
    is_rna = (aatype >= 21) & (aatype <= 25)

    pred_ca = pred_rigids.get_trans().cpu().numpy()
    gt_ca = gt_rigids.get_trans().cpu().numpy()

    # Kabsch alignment on CA/C4' positions
    pred_ca_aligned, R = kabsch_align(pred_ca, gt_ca)
    ca_rmsd = np.sqrt(np.mean((pred_ca_aligned - gt_ca) ** 2))
    rmsd_raw = np.sqrt(np.mean((pred_ca - gt_ca) ** 2))

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  CA/C4' RMSD (raw): {rmsd_raw:.2f} A")
    print(f"  CA/C4' RMSD (Kabsch): {ca_rmsd:.2f} A")
    print(f"{'='*60}")

    # Reconstruct full-atom structure by applying predicted rotation to GT local coords
    # Correct formula: undo GT rotation, apply predicted rotation, then translate
    print("Reconstructing full-atom structures...")
    pred_rot_mats = pred_rigids.get_rots().get_rot_mats().cpu().numpy()  # [N, 3, 3]
    gt_rot_mats = gt_rigids.get_rots().get_rot_mats().cpu().numpy()      # [N, 3, 3]

    N = gt_atom37.shape[0]
    pred_atom37 = np.zeros_like(gt_atom37)
    for i in range(N):
        if gt_atom37_mask[i].sum() == 0:
            continue
        # Local coordinates in GT frame: undo GT rotation first
        local_coords = (gt_atom37[i] - gt_ca[i]) @ gt_rot_mats[i]
        # Apply predicted rotation
        local_coords_rot = local_coords @ pred_rot_mats[i].T
        # Place at predicted CA/C4' position
        pred_atom37[i] = local_coords_rot + pred_ca[i]

    # Apply same Kabsch alignment to full-atom structure
    pred_atom37_aligned = pred_atom37.copy()
    pred_atom37_aligned = pred_atom37_aligned - pred_ca.mean(axis=0)
    pred_atom37_aligned = pred_atom37_aligned @ R.T
    pred_atom37_aligned = pred_atom37_aligned + gt_ca.mean(axis=0)

    # Save full-atom PDBs
    build_full_atom_pdb(
        gt_atom37, gt_atom37_mask, aatype_arr, is_rna_arr, chain_index_arr,
        residue_index_arr, os.path.join(out_dir, 'gt_9dcf.pdb'), pdb_id='GT9D'
    )
    build_full_atom_pdb(
        pred_atom37, gt_atom37_mask, aatype_arr, is_rna_arr, chain_index_arr,
        residue_index_arr, os.path.join(out_dir, 'pred_9dcf_raw.pdb'), pdb_id='PR9D'
    )
    build_full_atom_pdb(
        pred_atom37_aligned, gt_atom37_mask, aatype_arr, is_rna_arr, chain_index_arr,
        residue_index_arr, os.path.join(out_dir, 'pred_9dcf_aligned.pdb'), pdb_id='PA9D'
    )

    # Save raw coordinates as NPZ
    np.savez(
        os.path.join(out_dir, 'sampling_results_9dcf.npz'),
        gt_ca=gt_ca,
        pred_ca=pred_ca,
        pred_ca_aligned=pred_ca_aligned,
        gt_atom37=gt_atom37,
        pred_atom37=pred_atom37,
        pred_atom37_aligned=pred_atom37_aligned,
        atom37_mask=gt_atom37_mask,
        aatype=aatype_arr,
        rmsd_raw=rmsd_raw,
        rmsd_kabsch=ca_rmsd,
    )
    print(f"Saved raw coordinates: {os.path.join(out_dir, 'sampling_results_9dcf.npz')}")

    wandb.log({'result/ca_rmsd': ca_rmsd, 'result/ca_rmsd_raw': rmsd_raw})
    wandb.finish()


if __name__ == '__main__':
    main()
