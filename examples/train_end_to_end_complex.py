#!/usr/bin/env python3
"""End-to-end batched training on multi-chain protein-RNA complexes."""

import os
import sys
import argparse
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from openfold.utils import rigid_utils as ru
from openfold.utils.loss import torsion_angle_loss
from openfold.np import residue_constants as rc
import wandb
from datetime import datetime

from src.model.end_to_end_scorenet import EndToEndDyneTrionScoreNet
from src.data import se3_diffuser
from src.data import all_atom
from src.data.protenix_pipeline.pdb_to_protenix import preprocess_pdb_protenix_full
from src.data.protenix_pipeline.complex_dataset import ProtenixComplexDataset, collate_protenix_complex


def main():
    parser = argparse.ArgumentParser(description='Batched end-to-end training on multi-chain complexes')
    parser.add_argument('--features_dir', type=str, default=None, help='Directory containing .pt feature files')
    parser.add_argument('--features_paths', type=str, nargs='+', default=None, help='Explicit list of .pt feature files')
    parser.add_argument('--pdb_path', type=str, default=None, help='Path to single PDB/mmCIF to preprocess on-the-fly')
    parser.add_argument('--output_dir', type=str, default='outputs/end_to_end_complex', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=1000, help='Steps per epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--num_blocks', type=int, default=4, help='Number of IPA blocks')
    parser.add_argument('--pairformer_blocks', type=int, default=4, help='Number of PairFormer blocks')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--wandb_project', type=str, default='dyneTrion-complex', help='WandB project')
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Coordinate scale factor')
    args = parser.parse_args()

    if args.features_dir is None and args.features_paths is None and args.pdb_path is None:
        parser.error("Either --features_dir, --features_paths, or --pdb_path must be provided")

    # Set random seeds
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # ============================================================
    # Build dataset
    # ============================================================
    if args.features_paths is not None:
        pt_paths = args.features_paths
    elif args.features_dir is not None:
        pt_paths = sorted(glob.glob(os.path.join(args.features_dir, "*.pt")))
    else:
        pt_paths = []

    on_the_fly = args.pdb_path is not None and len(pt_paths) == 0

    if on_the_fly:
        print(f"Preprocessing {args.pdb_path} on-the-fly...", flush=True)
        features = preprocess_pdb_protenix_full(
            args.pdb_path,
            chain_id=None,
            device=device,
            use_msa=False,
        )
        dataset = ProtenixComplexDataset([features])
    else:
        if len(pt_paths) == 0:
            raise ValueError(f"No .pt files found in {args.features_dir}")
        print(f"Loading {len(pt_paths)} preprocessed complexes", flush=True)
        for p in pt_paths:
            print(f"  {p}")
        dataset = ProtenixComplexDataset(pt_paths)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_protenix_complex,
    )

    # ============================================================
    # WandB
    # ============================================================
    if not args.no_wandb:
        run_name = f"e2e_complex_{datetime.now().strftime('%m%d_%H%M')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                'lr': args.lr,
                'epochs': args.epochs,
                'steps_per_epoch': args.steps_per_epoch,
                'batch_size': args.batch_size,
                'num_blocks': args.num_blocks,
                'pairformer_blocks': args.pairformer_blocks,
                'scale_factor': args.scale_factor,
                'num_complexes': len(dataset),
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
        data_iter = iter(dataloader)

        for step in range(args.steps_per_epoch):
            model.train()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Move batch to device (handles nested dicts)
            def move_to_device(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.to(device)
                elif isinstance(obj, dict):
                    return {k: move_to_device(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [move_to_device(v) for v in obj]
                return obj

            batch = move_to_device(batch)

            aatype = batch['aatype']  # [B, N]
            mask = batch['mask'].to(device)  # [B, N]
            rigids_0 = batch['rigids_0']  # [B, N, 7]
            torsion_gt = batch['torsion_angles_sin_cos']  # [B, N, 7, 2]
            alt_torsion_gt = batch['alt_torsion_angles_sin_cos']  # [B, N, 7, 2]
            torsion_mask = batch['torsion_angles_mask']  # [B, N, 7]
            input_feature_dict = batch['input_feature_dict']
            B = aatype.shape[0]

            gt_rigids = ru.Rigid.from_tensor_7(rigids_0)
            min_t = 0.02
            t_scalar = np.random.rand() * (1.0 - min_t) + min_t
            t = torch.full((B,), t_scalar, device=device)

            diffused = diffuser.forward_marginal(gt_rigids, t_scalar, diffuse_mask=None)
            rigids_t_tensor = diffused['rigids_t']
            gt_rot_score = diffused['rot_score']
            gt_trans_score = diffused['trans_score']

            # End-to-end forward
            model_out = model(aatype, rigids_t_tensor, t, mask, input_feature_dict=input_feature_dict)

            pred_rot_score = model_out['pred_rot_score']
            pred_trans_score = model_out['pred_trans_score']
            pred_angles = model_out['angles']
            pred_rigids_tensor = model_out['rigids']

            # Translation loss
            pred_trans_score_mse = (pred_trans_score - gt_trans_score).pow(2) * mask[..., None]
            trans_score_loss = pred_trans_score_mse.sum() / (mask.sum() * 3 + 1e-10)

            gt_trans_x0 = rigids_0[..., 4:] * coord_scale
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
            if torsion_mask.sum() > 0:
                torsion_loss = torsion_angle_loss(
                    a=pred_angles,
                    a_gt=torsion_gt,
                    a_alt_gt=alt_torsion_gt,
                    mask=torsion_mask,
                ).mean() * torsion_w
            else:
                torsion_loss = torch.tensor(0.0, device=device)

            # Full-atom backbone + side chain loss (protein only)
            is_protein = (aatype < 20).all().item()
            bb_atom_loss = torch.tensor(0.0, device=device)
            dist_mat_loss = torch.tensor(0.0, device=device)
            if is_protein:
                try:
                    pred_rigids_obj = ru.Rigid.from_tensor_7(pred_rigids_tensor)
                    gt_rigids_bb = ru.Rigid.from_tensor_7(rigids_0.type(torch.float32))

                    gt_atom37, gt_atom37_mask, _, _ = all_atom.compute_backbone_atom37(
                        gt_rigids_bb, aatype, torsion_gt
                    )
                    pred_atom37, pred_atom37_mask, _, _ = all_atom.compute_backbone_atom37(
                        pred_rigids_obj, aatype, pred_angles
                    )

                    bb_atom_mse = (gt_atom37 - pred_atom37).pow(2) * gt_atom37_mask[..., None]
                    bb_atom_loss = bb_atom_mse.sum() / (gt_atom37_mask.sum() * 3 + 1e-10) * bb_w

                    gt_dist_mat = torch.cdist(gt_atom37[..., :3, :], gt_atom37[..., :3, :])
                    pred_dist_mat = torch.cdist(pred_atom37[..., :3, :], pred_atom37[..., :3, :])
                    dist_mat_mse = ((gt_dist_mat - pred_dist_mat).pow(2) * mask[:, None, :]).mean()
                    dist_mat_loss = dist_mat_mse * dist_w
                except Exception as e:
                    print(f"Skipping full-atom loss due to error: {e}")

            total_loss = trans_loss + rot_loss + torsion_loss + bb_atom_loss + dist_mat_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()
            trained_steps += 1

            if step % 10 == 0:
                log_dict = {
                    'loss/total': total_loss.item(),
                    'loss/trans': trans_loss.item(),
                    'loss/rot': rot_loss.item(),
                    'loss/torsion': torsion_loss.item(),
                    'loss/bb_atom': bb_atom_loss.item(),
                    'loss/dist_mat': dist_mat_loss.item(),
                    'lr': scheduler.get_last_lr()[0],
                    'epoch': epoch,
                    'step': trained_steps,
                }
                print(
                    f"Epoch {epoch} Step {step} | loss={total_loss.item():.4f} "
                    f"trans={trans_loss.item():.4f} rot={rot_loss.item():.4f} "
                    f"torsion={torsion_loss.item():.4f} bb={bb_atom_loss.item():.4f} "
                    f"dist={dist_mat_loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )
                if not args.no_wandb:
                    wandb.log(log_dict)

        avg_loss = epoch_loss / args.steps_per_epoch
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}", flush=True)

        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"model_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.output_dir, "model_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, best_path)
            print(f"New best model saved (loss={best_loss:.4f})")

    print("Training complete.")


if __name__ == '__main__':
    main()
