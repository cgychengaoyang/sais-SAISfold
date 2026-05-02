#!/usr/bin/env python3
"""End-to-end training from PDB: pdb -> preprocess -> train without precomputed embeddings."""

import os
import sys
import argparse

sys.path.insert(0, '/inspire/ssd/project/sais-bio/public/tzuhsiungyang/Projects/SAISfold')

import numpy as np
import torch
from omegaconf import OmegaConf
from openfold.utils import rigid_utils as ru
from openfold.utils.loss import torsion_angle_loss
from openfold.np import residue_constants as rc
import wandb
from datetime import datetime

from src.model.end_to_end_scorenet import EndToEndDyneTrionScoreNet
from src.data import se3_diffuser
from src.data import all_atom
from src.data.pdb_preprocessor import preprocess_pdb_for_training
from src.data.protenix_pipeline.pdb_to_protenix import preprocess_pdb_protenix_full


def main():
    parser = argparse.ArgumentParser(description='End-to-end training from PDB')
    parser.add_argument('--pdb_path', type=str, default=None, help='Path to input PDB file (required unless --features_path is given)')
    parser.add_argument('--chain_id', type=str, default=None, help='Chain ID to train on (default: first chain)')
    parser.add_argument('--output_dir', type=str, default='outputs/end_to_end_from_pdb', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=1000, help='Steps per epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_blocks', type=int, default=4, help='Number of IPA blocks')
    parser.add_argument('--pairformer_blocks', type=int, default=4, help='Number of PairFormer blocks')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--wandb_project', type=str, default='dyneTrion-protein', help='WandB project')
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Coordinate scale factor')
    parser.add_argument('--center', action='store_true', default=True, help='Center coordinates')
    parser.add_argument('--use_protenix_pipeline', action='store_true', help='Use full Protenix data pipeline with CCD features')
    parser.add_argument('--features_path', type=str, default=None, help='Path to preprocessed .pt features from preprocess_pdb_protenix.py')
    args = parser.parse_args()
    
    if args.pdb_path is None and args.features_path is None:
        parser.error("Either --pdb_path or --features_path must be provided")
    
    # Set random seeds
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    
    # ============================================================
    # Preprocess PDB
    # ============================================================
    print(f"Preprocessing PDB: {args.pdb_path}", flush=True)
    if args.chain_id:
        print(f"  Chain: {args.chain_id}", flush=True)
    
    if args.features_path is not None:
        print(f"Loading preprocessed features from {args.features_path}", flush=True)
        import pickle
        loaded = torch.load(args.features_path, map_location=device)
        # Unpickle atom_array / token_array if present
        for key in ['atom_array', 'token_array']:
            if key in loaded and isinstance(loaded[key], bytes):
                loaded[key] = pickle.loads(loaded[key])
        features = loaded
        input_feature_dict = features.get('input_feature_dict', None)
    elif args.use_protenix_pipeline:
        print("Using full Protenix pipeline with CCD features", flush=True)
        features = preprocess_pdb_protenix_full(
            args.pdb_path,
            chain_id=args.chain_id,
            device=device,
            use_msa=False,
        )
        input_feature_dict = features['input_feature_dict']
    else:
        print("Using lightweight PDB preprocessor", flush=True)
        features = preprocess_pdb_for_training(
            args.pdb_path,
            chain_id=args.chain_id,
            scale_factor=args.scale_factor,
            center=args.center,
            device=device,
        )
        input_feature_dict = None
    
    aatype_torch = features['aatype'].unsqueeze(0)  # [1, N]
    rigids_0 = features['rigids_0']
    torsion_gt_torch = features['torsion_angles_sin_cos'].unsqueeze(0)  # [1, N, 7, 2]
    alt_torsion_gt_torch = features['alt_torsion_angles_sin_cos'].unsqueeze(0)  # [1, N, 7, 2]
    torsion_mask_torch = features['torsion_angles_mask'].unsqueeze(0)  # [1, N, 7]
    seq_idx = features['seq_idx']  # [1, N]
    
    N = aatype_torch.shape[1]
    mask = torch.ones(1, N, device=device)
    B = args.batch_size
    
    print(f"Sequence length: {N}", flush=True)
    print(f"Torsion mask sum: {torsion_mask_torch.sum():.0f}/{N*7} angles available", flush=True)
    
    # ============================================================
    # WandB
    # ============================================================
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"e2e_pdb_{os.path.basename(args.pdb_path).split('.')[0]}_{datetime.now().strftime('%m%d_%H%M')}",
            config={
                'lr': args.lr,
                'epochs': args.epochs,
                'steps_per_epoch': args.steps_per_epoch,
                'batch_size': args.batch_size,
                'num_blocks': args.num_blocks,
                'pairformer_blocks': args.pairformer_blocks,
                'scale_factor': args.scale_factor,
                'pdb_path': args.pdb_path,
                'chain_id': args.chain_id,
                'end_to_end': True,
            }
        )
    
    # ============================================================
    # Model setup
    # ============================================================
    conf = OmegaConf.create({
        'diffuser': {
            'diffuse_rot': True,
            'diffuse_trans': True,
            'r3': {'min_b': 0.1, 'max_b': 1.5, 'coordinate_scaling': 0.1},
            'so3': {'num_omega': 1000, 'num_sigma': 1000, 'min_sigma': 0.1, 'max_sigma': 1.5, 'schedule': 'logarithmic', 'use_cached_score': True, 'cache_dir': '.cache'},
        },
        'model': {'c_s_input': 384, 'c_z_input': 128, 'c_s': 256, 'c_z': 128, 'num_blocks': args.num_blocks}
    })
    
    diffuser = se3_diffuser.SE3Diffuser(conf.diffuser)
    model = EndToEndDyneTrionScoreNet(
        c_s_input=384,
        c_z_input=128,
        c_s=256,
        c_z=128,
        num_blocks=args.num_blocks,
        pairformer_blocks=args.pairformer_blocks,
    ).to(device).float()
    model.set_diffuser(diffuser)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,}", flush=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-8, betas=(0.9, 0.95))
    warmup_steps = 100
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7)
        ],
        milestones=[warmup_steps]
    )
    
    # Loss weights
    coord_scale = 0.1
    rot_w = 0.5
    trans_w = 1.0
    bb_w = 1.0
    dist_w = 1.0
    torsion_w = 1.0
    aux_w = 0.5
    
    # ============================================================
    # Training loop
    # ============================================================
    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float('inf')
    trained_steps = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for step in range(args.steps_per_epoch):
            model.train()
            
            gt_rigids = ru.Rigid.from_tensor_7(rigids_0.unsqueeze(0))
            min_t = 0.02
            t = torch.rand(B, device=device) * (1.0 - min_t) + min_t
            
            diffused = diffuser.forward_marginal(gt_rigids, t[0].item(), diffuse_mask=None)
            rigids_t_tensor = diffused['rigids_t']
            gt_rot_score = diffused['rot_score']
            gt_trans_score = diffused['trans_score']
            
            # End-to-end forward
            model_out = model(aatype_torch, rigids_t_tensor, t, mask, seq_idx=seq_idx, input_feature_dict=input_feature_dict)
            
            pred_rot_score = model_out['pred_rot_score']
            pred_trans_score = model_out['pred_trans_score']
            pred_angles = model_out['angles']
            pred_rigids_tensor = model_out['rigids']
            
            # Translation loss
            pred_trans_score_mse = (pred_trans_score - gt_trans_score).pow(2)
            trans_score_loss = pred_trans_score_mse.sum() / (mask.sum() * 3 + 1e-10)
            
            gt_trans_x0 = rigids_0[..., 4:].unsqueeze(0) * coord_scale
            pred_trans_x0 = pred_rigids_tensor[..., 4:] * coord_scale
            trans_x0_mse = (gt_trans_x0 - pred_trans_x0).pow(2) * mask[..., None]
            trans_x0_loss = trans_x0_mse.sum() / (mask.sum() * 3 + 1e-10)
            
            trans_loss = (trans_score_loss + trans_x0_loss) * trans_w
            
            # Rotation loss
            gt_rot_angle = torch.norm(gt_rot_score, dim=-1, keepdim=True)
            gt_rot_axis = gt_rot_score / (gt_rot_angle + 1e-6)
            pred_rot_angle = torch.norm(pred_rot_score, dim=-1, keepdim=True)
            pred_rot_axis = pred_rot_score / (pred_rot_angle + 1e-6)
            axis_mse = (gt_rot_axis - pred_rot_axis).pow(2) * mask[..., None]
            angle_mse = (gt_rot_angle - pred_rot_angle).pow(2) * mask[..., None]
            axis_loss = axis_mse.sum() / (mask.sum() * 3 + 1e-10)
            angle_loss = angle_mse.sum() / (mask.sum() * 3 + 1e-10)
            rot_loss = (angle_loss + axis_loss) * rot_w
            
            # Torsion angle loss (skip for RNA/DNA where mask is all zeros)
            if torsion_mask_torch.sum() > 0:
                torsion_loss = torsion_angle_loss(
                    a=pred_angles,
                    a_gt=torsion_gt_torch,
                    a_alt_gt=alt_torsion_gt_torch,
                    mask=torsion_mask_torch,
                ) * torsion_w
            else:
                torsion_loss = torch.tensor(0.0, device=device)
            
            # Full-atom backbone + side chain loss (protein only)
            is_protein = (aatype_torch < 20).all().item()
            bb_atom_loss = torch.tensor(0.0, device=device)
            dist_mat_loss = torch.tensor(0.0, device=device)
            if is_protein:
                try:
                    pred_rigids_obj = ru.Rigid.from_tensor_7(pred_rigids_tensor)
                    gt_rigids_bb = ru.Rigid.from_tensor_7(rigids_0.unsqueeze(0).type(torch.float32))
                    
                    gt_atom37, gt_atom37_mask, _, _ = all_atom.compute_backbone_atom37(
                        gt_rigids_bb, aatype_torch, torsion_gt_torch
                    )
                    pred_atom37, pred_atom37_mask, _, _ = all_atom.compute_backbone_atom37(
                        pred_rigids_obj, aatype_torch, pred_angles
                    )
                    
                    bb_mask = pred_atom37_mask * mask[..., None]
                    bb_atom_loss = ((pred_atom37 - gt_atom37).pow(2) * bb_mask[..., None]).sum() / (bb_mask.sum() * 3 + 1e-10)
                    bb_atom_loss = bb_atom_loss * bb_w * aux_w
                    
                    gt_flat = gt_atom37.reshape(B, N * 37, 3)
                    pred_flat = pred_atom37.reshape(B, N * 37, 3)
                    gt_dists = torch.linalg.norm(gt_flat[:, :, None, :] - gt_flat[:, None, :, :], dim=-1)
                    pred_dists = torch.linalg.norm(pred_flat[:, :, None, :] - pred_flat[:, None, :, :], dim=-1)
                    flat_mask = torch.tile(mask[:, :, None], (1, 1, 37)).reshape(B, N * 37)
                    pair_mask = flat_mask[..., None] * flat_mask[:, None, :]
                    proximity = gt_dists < 6.0
                    pair_mask = pair_mask * proximity
                    dist_mat_loss = ((gt_dists - pred_dists).pow(2) * pair_mask).sum() / (pair_mask.sum() - N * 37 + 1e-10)
                    dist_mat_loss = dist_mat_loss * dist_w * aux_w
                except Exception as e:
                    print(f"BB loss error: {e}", flush=True)
                    bb_atom_loss = torch.tensor(0.0, device=device)
                    dist_mat_loss = torch.tensor(0.0, device=device)
            
            loss = rot_loss + trans_loss + torsion_loss + bb_atom_loss + dist_mat_loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Step {trained_steps}: NaN/Inf loss, skipping", flush=True)
                trained_steps += 1
                continue
            
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"Step {trained_steps}: Bad grad norm, skipping", flush=True)
                optimizer.zero_grad()
                trained_steps += 1
                continue
            
            optimizer.step()
            scheduler.step()
            trained_steps += 1
            
            loss_val = loss.item()
            epoch_loss += loss_val
            
            if trained_steps % 10 == 0 and not args.no_wandb:
                wandb.log({
                    'train/loss': loss_val,
                    'train/rot_loss': rot_loss.item(),
                    'train/trans_loss': trans_loss.item(),
                    'train/torsion_loss': torsion_loss.item(),
                    'train/bb_atom_loss': bb_atom_loss.item(),
                    'train/dist_mat_loss': dist_mat_loss.item(),
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'train/step': trained_steps,
                })
            
            if step % 100 == 0:
                print(f"  Epoch {epoch} Step {step}: loss={loss_val:.2f} rot={rot_loss.item():.2f} trans={trans_loss.item():.2f} tor={torsion_loss.item():.2f} bb={bb_atom_loss.item():.2f} dist={dist_mat_loss.item():.2f}", flush=True)
        
        avg_epoch_loss = epoch_loss / args.steps_per_epoch
        print(f"Epoch {epoch} avg loss: {avg_epoch_loss:.2f}", flush=True)
        if not args.no_wandb:
            wandb.log({'train/epoch_loss': avg_epoch_loss, 'epoch': epoch})
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            ckpt_path = f'{args.output_dir}/best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'trained_steps': trained_steps,
            }, ckpt_path)
            print(f"  Saved best model: {ckpt_path}", flush=True)
    
    print(f"\nTraining done. Best epoch loss: {best_loss:.2f}", flush=True)
    
    # ============================================================
    # Evaluation
    # ============================================================
    print("\nEvaluating...", flush=True)
    model.eval()
    with torch.no_grad():
        t_test = torch.tensor([0.5, 0.25, 0.1, 0.05], device=device)
        for t_val in t_test:
            rigids_t = diffuser.forward_marginal(ru.Rigid.from_tensor_7(rigids_0.unsqueeze(0)), t_val.item(), diffuse_mask=None)['rigids_t']
            out = model(aatype_torch, rigids_t, t_val.unsqueeze(0), mask, seq_idx=seq_idx, input_feature_dict=input_feature_dict)
            pred_ca = ru.Rigid.from_tensor_7(out['rigids']).get_trans()[0].cpu().numpy()
            gt_ca = rigids_0[..., 4:].cpu().numpy()
            pred_ca = pred_ca - pred_ca.mean(axis=0)
            gt_ca = gt_ca - gt_ca.mean(axis=0)
            H = pred_ca.T @ gt_ca
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            aligned = pred_ca @ R
            rmsd = np.sqrt(np.mean((aligned - gt_ca)**2))
            print(f"  Direct denoising t={t_val.item():.2f}: CA RMSD={rmsd:.2f}A", flush=True)
    
    # Reverse diffusion sampling with predicted angles
    print("\nReverse diffusion sampling (50 steps) with predicted torsion angles...", flush=True)
    with torch.no_grad():
        num_steps = 50
        min_t = 0.01
        reverse_steps = torch.linspace(min_t, 1.0, num_steps, device=device).flip(0)
        dt = (1.0 - min_t) / num_steps
        sqrt_dt = torch.sqrt(torch.tensor(dt, device=device))

        current_rigid = diffuser.forward_marginal(ru.Rigid.from_tensor_7(rigids_0.unsqueeze(0)), t=1.0, diffuse_mask=None)['rigids_t']
        if torch.is_tensor(current_rigid):
            current_rigid = ru.Rigid.from_tensor_7(current_rigid)

        z_rot = torch.randn(num_steps, B, N, 3, device=device)
        z_trans = torch.randn(num_steps, B, N, 3, device=device)

        for step_idx, t in enumerate(reverse_steps[:-1]):
            t_batch = t.unsqueeze(0).expand(B)
            rigids_t_tensor = current_rigid.to_tensor_7()
            model_out = model(aatype_torch, rigids_t_tensor, t_batch, mask, seq_idx=seq_idx, input_feature_dict=input_feature_dict)
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
        model_out = model(aatype_torch, rigids_t_tensor, t_batch, mask, seq_idx=seq_idx, input_feature_dict=input_feature_dict)
        final_rigids = ru.Rigid.from_tensor_7(model_out['rigids'])
        pred_angles = model_out['angles']
        
        if is_protein:
            pred_atom37, pred_atom37_mask, _, _ = all_atom.compute_backbone_atom37(
                final_rigids, aatype_torch, pred_angles
            )
            gt_atom37, gt_atom37_mask, _, _ = all_atom.compute_backbone_atom37(
                ru.Rigid.from_tensor_7(rigids_0.unsqueeze(0).type(torch.float32)), aatype_torch, torsion_gt_torch
            )
            
            pred_ca = pred_atom37[0, :, 1, :].cpu().numpy()
            gt_ca = gt_atom37[0, :, 1, :].cpu().numpy()
            
            pred_ca_c = pred_ca - pred_ca.mean(axis=0)
            gt_ca_c = gt_ca - gt_ca.mean(axis=0)
            H = pred_ca_c.T @ gt_ca_c
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            aligned = pred_ca_c @ R
            rmsd = np.sqrt(np.mean((aligned - gt_ca_c)**2))
            print(f"  Sampled CA RMSD: {rmsd:.2f}A", flush=True)
            
            pred_all = pred_atom37[0].cpu().numpy()
            gt_all = gt_atom37[0].cpu().numpy()
            m = pred_atom37_mask[0].cpu().numpy() > 0
            diff = (pred_all[m] - gt_all[m])
            full_rmsd = np.sqrt(np.mean(diff**2))
            print(f"  Sampled full-atom RMSD: {full_rmsd:.2f}A", flush=True)
        else:
            pred_ca = ru.Rigid.from_tensor_7(model_out['rigids']).get_trans()[0].cpu().numpy()
            gt_ca = rigids_0[..., 4:].cpu().numpy()
            pred_ca_c = pred_ca - pred_ca.mean(axis=0)
            gt_ca_c = gt_ca - gt_ca.mean(axis=0)
            H = pred_ca_c.T @ gt_ca_c
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            aligned = pred_ca_c @ R
            rmsd = np.sqrt(np.mean((aligned - gt_ca_c)**2))
            print(f"  Sampled CA RMSD: {rmsd:.2f}A", flush=True)
            print(f"  (Full-atom RMSD skipped for non-protein)", flush=True)
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
