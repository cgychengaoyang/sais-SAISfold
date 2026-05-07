#!/usr/bin/env python3
"""Setup correct 9DCF data for overfitting test.

Maps mmCIF residues to Protenix embedding tokens using asym_id and residue_index.
Extracts full backbone atoms for both protein and RNA.
"""
import os
import sys
import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser

from my_01_study.utils.root_utils import root_path_02

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openfold.np import residue_constants as rc

# Try to import RhoFold RNA constants
try:
    from rhofold.utils.constants import RNA_CONSTANTS
    HAS_RHOFOLD = True
except ImportError:
    HAS_RHOFOLD = False
    print("Warning: RhoFold not available")


def create_9dcf_data():
    # Paths
    embed_path = f'{root_path_02}/public/xiangwenkai/Protenix_v1/extract_embedding/output/embeddings_npz/9DCF_seed101.npz'
    cif_path = 'overfitting_data/9DCF/9dcf.cif'
    output_dir = 'overfitting_data/9DCF'
    os.makedirs(output_dir, exist_ok=True)

    # Load embedding metadata
    embed_data = np.load(embed_path)
    residue_index = embed_data['residue_index'].astype(int)
    asym_id = embed_data['asym_id'].astype(int)
    N = len(residue_index)
    print(f"Embedding tokens: {N}")
    print(f"asym_id unique: {np.unique(asym_id)}")
    for a in np.unique(asym_id):
        mask = asym_id == a
        print(f"  asym_id {a}: {mask.sum()} tokens, residue_index {residue_index[mask].min()}-{residue_index[mask].max()}")

    # Parse mmCIF
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('9dcf', cif_path)
    model = list(structure.get_models())[0]

    # Map mmCIF chains to asym_id
    chain_ids = [c.id for c in model.get_chains()]
    print(f"mmCIF chains: {chain_ids}")

    # Initialize arrays
    all_atom_positions = np.zeros((N, 37, 3), dtype=np.float32)
    all_atom_mask = np.zeros((N, 37), dtype=np.float32)
    aatype = np.zeros(N, dtype=np.int64)
    is_rna = np.zeros(N, dtype=bool)
    restype_protenix = np.zeros(N, dtype=np.int64)
    chain_index = np.zeros(N, dtype=np.int64)

    # Mappings
    AA_TO_AATYPE = rc.restype_order
    RNA_RESTYPE_MAP = {'A': 21, 'G': 22, 'C': 23, 'U': 24, 'N': 25}

    # For each chain in mmCIF, map to asym_id
    for chain_idx, chain_id in enumerate(chain_ids):
        chain = model[chain_id]
        residues = [r for r in chain if r.id[0] == ' ']  # skip HETATM/waters
        print(f"Chain {chain_id} (asym_id {chain_idx}): {len(residues)} residues")

        for res in residues:
            res_seq = res.id[1]
            res_name = res.resname.strip()

            # Find the token index matching this residue
            token_mask = (asym_id == chain_idx) & (residue_index == res_seq)
            if not token_mask.any():
                print(f"  Warning: no token found for chain {chain_id} residue {res_seq} {res_name}")
                continue

            token_idx = np.where(token_mask)[0][0]
            chain_index[token_idx] = chain_idx

            if res_name in rc.restype_3to1:
                # Protein
                aa = rc.restype_3to1[res_name]
                aatype[token_idx] = AA_TO_AATYPE.get(aa, 20)
                restype_protenix[token_idx] = aatype[token_idx]
                is_rna[token_idx] = False

                # Extract protein atoms at correct atom37 positions
                for atom_name in rc.residue_atoms.get(res_name, []):
                    if atom_name in res and atom_name in rc.atom_order:
                        atom_idx = rc.atom_order[atom_name]
                        all_atom_positions[token_idx, atom_idx] = res[atom_name].get_coord()
                        all_atom_mask[token_idx, atom_idx] = 1.0
            else:
                # RNA
                base = res_name[-1] if res_name.startswith('D') and len(res_name) == 2 else res_name
                aatype[token_idx] = RNA_RESTYPE_MAP.get(base, 25)
                restype_protenix[token_idx] = aatype[token_idx]
                is_rna[token_idx] = True

                if HAS_RHOFOLD:
                    # Place RNA atoms using RhoFold ordering
                    atom_names = RNA_CONSTANTS.ATOM_NAMES_PER_RESD.get(res_name, [])
                    for j, atom_name in enumerate(atom_names):
                        if j >= 37:
                            break
                        if atom_name in res:
                            all_atom_positions[token_idx, j] = res[atom_name].get_coord()
                            all_atom_mask[token_idx, j] = 1.0
                else:
                    # Fallback: place key atoms
                    key_atoms = {"C4'": 0, "C1'": 1}
                    base_n = 'N9' if base in ('A', 'G') else 'N1'
                    key_atoms[base_n] = 2
                    key_atoms['P'] = 19
                    for atom_name, idx in key_atoms.items():
                        if atom_name in res:
                            all_atom_positions[token_idx, idx] = res[atom_name].get_coord()
                            all_atom_mask[token_idx, idx] = 1.0

    # Set chain_index for all tokens based on asym_id
    chain_index = asym_id.copy()

    # Create bb_mask (backbone atoms)
    bb_mask = np.zeros((N,), dtype=bool)
    for i in range(N):
        if is_rna[i]:
            # RNA: need C4' (idx 0) and C1' (idx 1) for frame construction
            # atom37_to_frames uses indices 2,1,0 as C,CA,N -> for RNA we want:
            # index 0 = C4', index 1 = C1', index 2 = N9/N1
            bb_mask[i] = (all_atom_mask[i, 0] > 0) and (all_atom_mask[i, 1] > 0) and (all_atom_mask[i, 2] > 0)
        else:
            # Protein: N, CA, C
            bb_mask[i] = (all_atom_mask[i, 0] > 0) and (all_atom_mask[i, 1] > 0) and (all_atom_mask[i, 2] > 0)

    # Print stats
    mapped = (all_atom_mask.sum(axis=1) > 0).sum()
    print(f"\nMapped residues with atoms: {mapped}/{N}")
    print(f"Protein residues: {(~is_rna).sum()}")
    print(f"RNA residues: {is_rna.sum()}")
    print(f"Backbone-ready residues: {bb_mask.sum()}/{N}")

    # Save structure.npz
    struct_path = os.path.join(output_dir, 'structure.npz')
    np.savez(struct_path,
        all_atom_positions=all_atom_positions,
        all_atom_mask=all_atom_mask,
        aatype=aatype,
        is_rna=is_rna,
        restype_protenix=restype_protenix,
        residue_index=residue_index,
        chain_index=chain_index,
        bb_mask=bb_mask,
    )
    print(f"\nSaved structure.npz to {struct_path}")

    # Trajectory format expected by DyneTrion_data_loader_dynamic
    traj_path = os.path.join(output_dir, 'trajectory.npz')
    # One-hot aatype
    aatype_onehot = np.zeros((N, 22), dtype=np.float32)
    for i in range(N):
        if aatype[i] < 22:
            aatype_onehot[i, aatype[i]] = 1.0
        else:
            aatype_onehot[i, 20] = 1.0  # UNK for out-of-range (shouldn't happen)

    # Expand to 5 frames (standard for trajectory.npz)
    all_atom_positions_frames = np.tile(all_atom_positions[np.newaxis, ...], (5, 1, 1, 1))

    np.savez(traj_path,
        all_atom_positions=all_atom_positions_frames,
        all_atom_mask=all_atom_mask,
        aatype=aatype_onehot,
        is_rna=is_rna,
        restype_protenix=restype_protenix,
        residue_index=residue_index,
        chain_index=chain_index,
        bb_mask=bb_mask,
    )
    print(f"Saved trajectory.npz to {traj_path}")

    # Copy/symlink embedding
    import shutil
    embed_dst = os.path.join(output_dir, 'embedding.npz')
    shutil.copy(embed_path, embed_dst)
    print(f"Copied embedding to {embed_dst}")

    # Create CSV
    csv_path = os.path.join(output_dir, '9dcf_data.csv')
    df = pd.DataFrame({
        'pdb_id': ['9DCF'],
        'pos_path': [traj_path],
        'embed_path': [embed_dst],
        'seq': ['MET'],  # dummy seq to avoid NaN drop
        'total_seq_len': [N],
        'seq_len': [[N]],
    })
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    # Verify loading
    test = np.load(struct_path)
    print(f"\nVerification:")
    print(f"  all_atom_positions: {test['all_atom_positions'].shape}")
    print(f"  aatype: {test['aatype'].shape}")
    print(f"  is_rna: {test['is_rna'].sum()} True")
    print(f"  bb_mask: {test['bb_mask'].sum()} True")


if __name__ == '__main__':
    create_9dcf_data()
