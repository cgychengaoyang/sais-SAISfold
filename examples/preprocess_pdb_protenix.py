#!/usr/bin/env python3
"""Preprocess PDB file using full Protenix pipeline and save features for training."""

import os
import sys
import argparse
import pickle

sys.path.insert(0, '/inspire/ssd/project/sais-bio/public/tzuhsiungyang/Projects/SAISfold')

import torch

from src.data.protenix_pipeline.pdb_to_protenix import preprocess_pdb_protenix_full


def main():
    parser = argparse.ArgumentParser(description='Preprocess PDB with Protenix pipeline')
    parser.add_argument('--pdb_path', type=str, required=True, help='Path to input PDB file')
    parser.add_argument('--chain_id', type=str, default=None, help='Chain ID to use')
    parser.add_argument('--output_path', type=str, default=None, help='Output .pt file path')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--use_msa', action='store_true', help='Use MSA (requires DBs)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    if args.output_path is None:
        base = os.path.splitext(os.path.basename(args.pdb_path))[0]
        chain = args.chain_id if args.chain_id else 'all'
        args.output_path = f"{base}_chain{chain}_protenix_features.pt"

    print(f"Preprocessing {args.pdb_path} (chain={args.chain_id})...", flush=True)
    result = preprocess_pdb_protenix_full(
        args.pdb_path,
        chain_id=args.chain_id,
        device=device,
        use_msa=args.use_msa,
    )

    # Move everything to CPU for saving
    save_dict = {}
    for key, val in result.items():
        if isinstance(val, torch.Tensor):
            save_dict[key] = val.cpu()
        elif key == 'input_feature_dict':
            # Save tensors in feature dict, leave numpy arrays as-is
            feature_dict = {}
            for k, v in val.items():
                if isinstance(v, torch.Tensor):
                    feature_dict[k] = v.cpu()
                else:
                    feature_dict[k] = v
            save_dict[key] = feature_dict
        elif key in ('atom_array', 'token_array'):
            # Biotite AtomArray can't be pickled directly by torch.save
            # Use pickle for these
            save_dict[key] = pickle.dumps(val)
        else:
            save_dict[key] = val

    torch.save(save_dict, args.output_path)
    print(f"Saved to {args.output_path}", flush=True)
    print(f"  Sequence length: {save_dict['aatype'].shape[0]}", flush=True)
    print(f"  Num atoms: {save_dict['input_feature_dict']['ref_pos'].shape[0]}", flush=True)
    print(f"  Torsion mask sum: {save_dict['torsion_angles_mask'].sum():.0f}", flush=True)


if __name__ == '__main__':
    main()
