#!/usr/bin/env python3
"""Multimer-style direct training for 9DCF on SAISfold corrected data."""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
os.environ['PYTHONUNBUFFERED'] = '1'

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from omegaconf import OmegaConf
from openfold.utils import rigid_utils as ru
from openfold.utils.rigid_utils import Rotation

from src.model import score_based_ipa
from src.data import se3_diffuser

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}", flush=True)

# Load corrected 9DCF data
data_dir = 'overfitting_data/9DCF'
embed_data = dict(np.load(os.path.join(data_dir, 'embedding.npz')))
struct_data = dict(np.load(os.path.join(data_dir, 'structure.npz')))

node_repr = torch.from_numpy(embed_data['single_s']).float().to(device)
edge_repr = torch.from_numpy(embed_data['pair_z']).float().to(device)
all_atom_pos = torch.from_numpy(struct_data['all_atom_positions']).float()
all_atom_mask = torch.from_numpy(struct_data['all_atom_mask']).float()
aatype = torch.from_numpy(struct_data['aatype']).long()

# Normalize embeddings like multimer does
node_std = node_repr.std()
if node_std > 1.0:
    print(f"Normalizing node_repr by std={node_std:.2f}")
    node_repr = node_repr / node_std
edge_std = edge_repr.std()
if edge_std > 1.0:
    print(f"Normalizing edge_repr by std={edge_std:.2f}")
    edge_repr = edge_repr / edge_std

N = node_repr.shape[0]
print(f"9DCF: {N} tokens", flush=True)

# Build backbone mask
bb_mask = torch.from_numpy(struct_data['bb_mask']).bool()
print(f"Backbone-ready residues: {bb_mask.sum().item()}/{N}", flush=True)

# Build rigids from backbone
n_pos = all_atom_pos[:, 0, :]
ca_pos = all_atom_pos[:, 1, :]
c_pos = all_atom_pos[:, 2, :]

# Fill missing N/C with CA for frame construction
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

diffuse_mask = bb_mask.unsqueeze(0).float().to(device)

node_s = node_repr.unsqueeze(0)
edge_z = edge_repr.unsqueeze(0)

# Config
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

num_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {num_params:,}", flush=True)

# Try warm-start from existing SAISfold checkpoint if available
init_ckpt = 'outputs/overfit_9dcf/final_model.pth'
if os.path.exists(init_ckpt):
    print(f"Warm-starting from {init_ckpt}")
    ckpt = torch.load(init_ckpt, map_location=device, weights_only=False)
    # The checkpoint is a raw state dict
    if isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        state = ckpt
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    # Filter keys for DyneTrionScoreNet vs FullScoreNetwork
    model_state = model.state_dict()
    filtered = {}
    for k, v in state.items():
        key = k[7:] if k.startswith('module.') else k
        if key in model_state and v.shape == model_state[key].shape:
            filtered[key] = v
    model.load_state_dict(filtered, strict=False)
    print(f"Loaded {len(filtered)} tensors")
else:
    print("Training from scratch")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

output_dir = 'outputs/9dcf_multimer_style'
os.makedirs(output_dir, exist_ok=True)

num_epochs = 5000
best_loss = float('inf')

print("Starting training...", flush=True)
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    t = torch.rand(1, device=device) * 0.999 + 0.001
    
    gt_rigids = ru.Rigid.from_tensor_7(rigids_0.unsqueeze(0))
    marginal = diffuser.forward_marginal(
        gt_rigids,
        t.item(),
        diffuse_mask=diffuse_mask,
    )
    rigids_t = marginal['rigids_t']

    out = model(node_s, edge_z, rigids_t, t, mask)

    pred_rot = ru.Rigid.from_tensor_7(out['rigids']).get_rots().get_rot_mats()
    gt_rot = gt_rigids.get_rots().get_rot_mats()
    rot_mse = (pred_rot - gt_rot) ** 2
    rot_loss = (rot_mse * mask[..., None, None]).sum() / (mask.sum() * 9 + 1e-10)

    pred_trans = ru.Rigid.from_tensor_7(out['rigids']).get_trans()
    gt_trans = rigids_0[..., 4:].unsqueeze(0)
    trans_mse = (pred_trans - gt_trans) ** 2
    trans_loss = (trans_mse * mask[..., None]).sum() / (mask.sum() * 3 + 1e-10)

    loss = rot_loss + trans_loss

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Epoch {epoch}: NaN/Inf loss detected! Skipping.")
        continue

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
        print(f"Epoch {epoch}: Bad grad norm, skipping")
        optimizer.zero_grad()
        continue

    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f} rot={rot_loss.item():.4f} trans={trans_loss.item():.4f} grad={grad_norm:.2f}", flush=True)

    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, os.path.join(output_dir, 'best_model.pth'))

# Final evaluation
print("\nEvaluating direct denoising...", flush=True)
model.eval()
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

print(f"\nTraining complete. Best loss: {best_loss:.4f}")
print(f"Checkpoint saved to {output_dir}/best_model.pth")
