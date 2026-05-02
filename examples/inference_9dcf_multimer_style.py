#!/usr/bin/env python3
"""Inference for multimer-style 9DCF model — saves proper full-atom PDB."""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import re
from omegaconf import OmegaConf
from openfold.utils import rigid_utils as ru
from openfold.utils.rigid_utils import Rotation
from Bio.PDB import Structure, Model, Chain, Residue, Atom, PDBIO

from src.model import score_based_ipa
from src.data import se3_diffuser

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}", flush=True)

try:
    from rhofold.utils.constants import RNA_CONSTANTS
    HAS_RHOFOLD = True
except ImportError:
    HAS_RHOFOLD = False

# Load data
data_dir = 'overfitting_data/9DCF'
embed_data = dict(np.load(os.path.join(data_dir, 'embedding.npz')))
struct_data = dict(np.load(os.path.join(data_dir, 'structure.npz')))

node_repr = torch.from_numpy(embed_data['single_s']).float().to(device)
edge_repr = torch.from_numpy(embed_data['pair_z']).float().to(device)
all_atom_pos = torch.from_numpy(struct_data['all_atom_positions']).float()
all_atom_mask = torch.from_numpy(struct_data['all_atom_mask']).float()
aatype = torch.from_numpy(struct_data['aatype']).long()
is_rna = struct_data['is_rna']
chain_index = struct_data['chain_index']
residue_index = struct_data['residue_index']
bb_mask = torch.from_numpy(struct_data['bb_mask']).bool()

N = node_repr.shape[0]
node_std = node_repr.std()
if node_std > 1.0:
    node_repr = node_repr / node_std
edge_std = edge_repr.std()
if edge_std > 1.0:
    edge_repr = edge_repr / edge_std

n_pos = all_atom_pos[:, 0, :]
ca_pos = all_atom_pos[:, 1, :]
c_pos = all_atom_pos[:, 2, :]
n_pos_filled = n_pos.clone()
c_pos_filled = c_pos.clone()
n_pos_filled[~bb_mask] = ca_pos[~bb_mask]
c_pos_filled[~bb_mask] = ca_pos[~bb_mask]

v1 = (n_pos_filled - ca_pos)
v1 = v1 / (v1.norm(dim=-1, keepdim=True) + 1e-7)
v2 = (c_pos_filled - ca_pos)
v2 = v2 / (v2.norm(dim=-1, keepdim=True) + 1e-7)
e1 = v2
e3 = torch.cross(v1, v2, dim=-1)
e3 = e3 / (e3.norm(dim=-1, keepdim=True) + 1e-7)
e2 = torch.cross(e3, e1, dim=-1)
rotmat = torch.stack([e1, e2, e3], dim=-1)

quat = Rotation(rot_mats=rotmat).get_quats()
rigids_0 = torch.cat([quat, ca_pos], dim=-1).to(device)
mask = bb_mask.unsqueeze(0).to(device).float()

node_s = node_repr.unsqueeze(0)
edge_z = edge_repr.unsqueeze(0)
B = 1

conf = OmegaConf.create({
    'diffuser': {
        'diffuse_rot': True,
        'diffuse_trans': True,
        'r3': {'min_b': 0.1, 'max_b': 1.5, 'coordinate_scaling': 0.1},
        'so3': {'num_omega': 1000, 'num_sigma': 1000, 'min_sigma': 0.1, 'max_sigma': 1.5,
                'schedule': 'logarithmic', 'use_cached_score': True, 'cache_dir': '.cache'},
    },
    'model': {'c_s_input': 384, 'c_z_input': 128, 'c_s': 256, 'c_z': 128, 'num_blocks': 4}
})

diffuser = se3_diffuser.SE3Diffuser(conf.diffuser)
model = score_based_ipa.DyneTrionScoreNet(384, 128, 256, 128, 4).to(device).float()
model.set_diffuser(diffuser)

ckpt_path = 'outputs/9dcf_multimer_style/best_model.pth'
print(f"Loading checkpoint from {ckpt_path}", flush=True)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

output_dir = 'outputs/9dcf_multimer_style'
os.makedirs(output_dir, exist_ok=True)

# Direct denoising evaluation
print("\nEvaluating direct denoising...", flush=True)
with torch.no_grad():
    t_test = torch.tensor([0.5, 0.25, 0.1, 0.05], device=device)
    for t_val in t_test:
        rigids_t = diffuser.forward_marginal(
            ru.Rigid.from_tensor_7(rigids_0.unsqueeze(0)),
            t_val.item(),
            diffuse_mask=None
        )['rigids_t']
        out = model(node_s, edge_z, rigids_t, t_val.unsqueeze(0), mask)
        pred_ca = ru.Rigid.from_tensor_7(out['rigids']).get_trans()[0].cpu().numpy()
        gt_ca = rigids_0[..., 4:].cpu().numpy()
        valid = bb_mask.cpu().numpy()
        pred_ca_c = pred_ca[valid] - pred_ca[valid].mean(axis=0)
        gt_ca_c = gt_ca[valid] - gt_ca[valid].mean(axis=0)
        H = pred_ca_c.T @ gt_ca_c
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        aligned = pred_ca_c @ R
        rmsd = np.sqrt(np.mean((aligned - gt_ca_c)**2))
        print(f"  t={t_val.item():.2f}: CA RMSD={rmsd:.2f}A", flush=True)

# Reverse diffusion sampling
print("\nReverse diffusion sampling (50 steps)...", flush=True)
with torch.no_grad():
    num_steps = 50
    min_t = 0.01
    reverse_steps = torch.linspace(min_t, 1.0, num_steps, device=device).flip(0)
    dt = (1.0 - min_t) / num_steps
    sqrt_dt = torch.sqrt(torch.tensor(dt, device=device))

    current_rigid_tensor = diffuser.forward_marginal(
        ru.Rigid.from_tensor_7(rigids_0.unsqueeze(0)), t=1.0, diffuse_mask=None
    )['rigids_t']
    current_rigid = ru.Rigid.from_tensor_7(current_rigid_tensor)

    z_rot = torch.randn(num_steps, B, N, 3, device=device)
    z_trans = torch.randn(num_steps, B, N, 3, device=device)

    for step_idx, t in enumerate(reverse_steps[:-1]):
        t_batch = t.unsqueeze(0).expand(B)
        rigids_t_tensor = current_rigid.to_tensor_7()
        model_out = model(node_s, edge_z, rigids_t_tensor, t_batch, mask)
        current_rigid = diffuser.reverse(
            rigid_t=current_rigid,
            rot_score=model_out['pred_rot_score'],
            trans_score=model_out['pred_trans_score'],
            t=t, dt=dt, sqrt_dt=sqrt_dt,
            z_rot=z_rot[step_idx], z_trans=z_trans[step_idx],
            diffuse_mask=mask, center=True, noise_scale=1.0, device=device,
        )

    t_batch = reverse_steps[-1].unsqueeze(0).expand(B)
    rigids_t_tensor = current_rigid.to_tensor_7()
    model_out = model(node_s, edge_z, rigids_t_tensor, t_batch, mask)
    final_rigids = ru.Rigid.from_tensor_7(model_out['rigids'])

    pred_ca = final_rigids.get_trans()[0].cpu().numpy()
    gt_ca = rigids_0[..., 4:].cpu().numpy()
    valid = bb_mask.cpu().numpy()

    pred_ca_c = pred_ca[valid] - pred_ca[valid].mean(axis=0)
    gt_ca_c = gt_ca[valid] - gt_ca[valid].mean(axis=0)
    H = pred_ca_c.T @ gt_ca_c
    U, S, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T
    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1
        R_align = Vt.T @ U.T
    aligned = pred_ca_c @ R_align
    rmsd = np.sqrt(np.mean((aligned - gt_ca_c)**2))
    print(f"  Sampled CA RMSD: {rmsd:.2f}A", flush=True)

# Save PDBs
print("\nSaving PDBs...", flush=True)

PROTEIN_AA_1TO3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}
RNA_RESNAME_MAP = {21: '  A', 22: '  G', 23: '  C', 24: '  U', 25: '  N'}
RNA_BASE_MAP = {21: 'A', 22: 'G', 23: 'C', 24: 'U', 25: 'N'}

def get_res_name(aa_idx, is_rna_flag):
    if is_rna_flag:
        return RNA_RESNAME_MAP.get(int(aa_idx), '  N')[:3]
    if 0 <= int(aa_idx) < 20:
        from openfold.np import residue_constants as rc
        return PROTEIN_AA_1TO3[rc.restypes[int(aa_idx)]]
    return 'UNK'

def get_element(atom_name):
    m = re.match(r'([A-Za-z]+)', str(atom_name))
    if not m:
        return 'C'
    elem = m.group(1).upper()
    if elem in ('CL', 'FE', 'BR', 'ZN', 'CA', 'MG', 'NA', 'CU', 'MN'):
        return elem
    return elem[0]

def build_pdb(atom_pos, atom_mask, aatype, is_rna, chain_index, residue_index, out_path, pdb_id='PRED'):
    structure = Structure.Structure(pdb_id)
    model_obj = Model.Model(0)
    structure.add(model_obj)
    chains = {}
    for cid in np.unique(chain_index):
        chains[int(cid)] = Chain.Chain(chr(ord('A') + int(cid)))
        model_obj.add(chains[int(cid)])
    serial = 1
    for i in range(N):
        if atom_mask[i].sum() == 0:
            continue
        cid = int(chain_index[i])
        res_num = int(residue_index[i])
        res_name = get_res_name(aatype[i], is_rna[i])
        residue = Residue.Residue((' ', res_num, ' '), res_name, 0)
        chains[cid].add(residue)
        if is_rna[i] and HAS_RHOFOLD:
            base = RNA_BASE_MAP.get(int(aatype[i]), 'N')
            atom_names = RNA_CONSTANTS.ATOM_NAMES_PER_RESD.get(base, [])
            for j, name in enumerate(atom_names):
                if j >= 37:
                    break
                if atom_mask[i, j] > 0:
                    atom = Atom.Atom(name, atom_pos[i, j], 0.0, 1.0, ' ', name, serial, get_element(name))
                    residue.add(atom)
                    serial += 1
        else:
            from openfold.np import residue_constants as rc
            for j in range(37):
                if atom_mask[i, j] > 0:
                    name = rc.atom_types[j]
                    atom = Atom.Atom(name, atom_pos[i, j], 0.0, 1.0, ' ', name, serial, get_element(name))
                    residue.add(atom)
                    serial += 1
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path))
    print(f"Saved: {out_path}")

# GT
gt_atom37 = all_atom_pos.numpy()
build_pdb(gt_atom37, all_atom_mask, aatype, is_rna, chain_index, residue_index,
          os.path.join(output_dir, 'gt.pdb'), 'GT')

# Sampled pred with full-atom reconstruction
pred_rot_mats = final_rigids.get_rots().get_rot_mats()[0].cpu().numpy()
gt_rot_mats = ru.Rigid.from_tensor_7(rigids_0.unsqueeze(0)).get_rots().get_rot_mats()[0].cpu().numpy()

pred_atom37 = np.zeros_like(gt_atom37)
for i in range(N):
    if all_atom_mask[i].sum() == 0:
        continue
    local_coords = (gt_atom37[i] - gt_ca[i]) @ gt_rot_mats[i]
    local_coords_rot = local_coords @ pred_rot_mats[i].T
    pred_atom37[i] = local_coords_rot + pred_ca[i]

# Apply Kabsch alignment to full-atom structure
pred_atom37_aligned = pred_atom37.copy()
pred_atom37_aligned = pred_atom37_aligned - pred_ca.mean(axis=0)
pred_atom37_aligned = pred_atom37_aligned @ R_align.T
pred_atom37_aligned = pred_atom37_aligned + gt_ca.mean(axis=0)

build_pdb(pred_atom37_aligned, all_atom_mask, aatype, is_rna, chain_index, residue_index,
          os.path.join(output_dir, 'pred_aligned.pdb'), 'PRED')

print("\nDone.")
