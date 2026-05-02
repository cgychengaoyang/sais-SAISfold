#!/usr/bin/env python3
"""Prepare RNA training data for DyneTrion from mmCIF and Protenix embeddings.

This script creates proper trajectory data compatible with DyneTrion's PdbDataset
for RNA structures, handling:
- mmCIF parsing and RNA structure extraction
- All RNA atoms (not just C1')
- Protenix-compatible restype encoding (A=22, C=23, G=24, U=25)
- HybridConverter-compatible all-atom representation

Usage:
    python prepare_rna_training_data.py --pdb_id 2KMJ --output_dir rna_data
    
    # With custom paths:
    python prepare_rna_training_data.py \
        --pdb_id 2KMJ \
        --cif_path /path/to/2KMJ.cif \
        --embed_path /path/to/2KMJ_seed101.npz \
        --output_dir rna_data
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
import requests
from Bio.PDB import MMCIFParser, PDBIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# RNA/DNA nucleotide to Protenix restype encoding
# Protenix uses AlphaFold3 encoding:
# Protein: 0-20 (20 AA + UNK=20)
# RNA: 21-25 (A=21, G=22, C=23, U=24, N=25)
# DNA: 26-30 (DA=26, DG=27, DC=28, DT=29, DN=30)
# Gap: 31
RNA_RESTYPE_MAP = {
    'A': 21,
    'G': 22,
    'C': 23,
    'U': 24,
    'N': 25,
    'DA': 26,  # Deoxyadenosine (handle DNA)
    'DG': 27,
    'DC': 28,
    'DT': 29,
    'DN': 30,
}

# Standard RNA atom names per residue type (from RhoFold constants)
# Order matches the expected atom ordering in RNA all-atom representations
RNA_ATOM_NAMES = {
    'A': ["C4'", "C1'", 'N9', "C2'", "C3'", "C5'", "O2'", "O3'", "O4'", 
          'N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N6', 'N7', 'C8', "O5'", 'P', 'OP1', 'OP2'],
    'G': ["C4'", "C1'", 'N9', "C2'", "C3'", "C5'", "O2'", "O3'", "O4'",
          'N1', 'N2', 'N3', 'C2', 'C4', 'C5', 'C6', 'N7', 'C8', 'O6', "O5'", 'P', 'OP1', 'OP2'],
    'U': ["C4'", "C1'", 'N1', "C2'", "C3'", "C5'", "O2'", "O3'", "O4'",
          'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6', "O5'", 'P', 'OP1', 'OP2'],
    'C': ["C4'", "C1'", 'N1', "C2'", "C3'", "C5'", "O2'", "O3'", "O4'",
          'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6', "O5'", 'P', 'OP1', 'OP2'],
}

# Atom name mapping from mmCIF/PDB format to internal format
# Handles quoted atom names like "O5'" vs O5'
ATOM_NAME_MAP = {
    "O5'": "O5'",
    "C5'": "C5'",
    "C4'": "C4'",
    "O4'": "O4'",
    "C3'": "C3'",
    "O3'": "O3'",
    "C2'": "C2'",
    "O2'": "O2'",
    "C1'": "C1'",
    'P': 'P',
    'OP1': 'OP1',
    'OP2': 'OP2',
    'OP3': 'OP3',
    'N9': 'N9',
    'N1': 'N1',
    'N2': 'N2',
    'N3': 'N3',
    'N4': 'N4',
    'N6': 'N6',
    'N7': 'N7',
    'C2': 'C2',
    'C4': 'C4',
    'C5': 'C5',
    'C6': 'C6',
    'C8': 'C8',
    'O2': 'O2',
    'O4': 'O4',
    'O6': 'O6',
}


def download_cif(pdb_id: str, output_path: str) -> str:
    """Download mmCIF file from RCSB PDB.
    
    Args:
        pdb_id: PDB ID (e.g., '2KMJ')
        output_path: Path to save the CIF file
        
    Returns:
        Path to downloaded file
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    response = requests.get(url)
    response.raise_for_status()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(response.text)
    
    print(f"Downloaded {pdb_id}.cif from RCSB")
    return output_path


def parse_rna_cif(cif_path: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Parse RNA structure from mmCIF file.
    
    Extracts all RNA atoms with proper naming and coordinates.
    
    Args:
        cif_path: Path to mmCIF file
        
    Returns:
        Tuple of (sequence_list, atom_data_dict)
        - sequence_list: List of nucleotide one-letter codes
        - atom_data_dict: Dictionary with keys:
            - 'atom_positions': [num_res, 37, 3] array of atom coordinates
            - 'atom_mask': [num_res, 37] array of atom presence mask
            - 'residue_index': [num_res] array of residue indices
            - 'chain_index': [num_res] array of chain indices
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('rna', cif_path)
    
    # Extract the first model (NMR structures have multiple models)
    model = list(structure)[0]
    
    residues = []
    residue_indices = []
    chain_indices = []
    
    for chain_idx, chain in enumerate(model):
        for residue in chain:
            res_name = residue.get_resname().strip()
            
            # Check if it's a nucleotide we can handle
            if res_name in RNA_RESTYPE_MAP:
                # Get residue index (handling insertion codes)
                res_id = residue.get_id()
                res_seq = res_id[1]
                
                residues.append({
                    'name': res_name[-1] if res_name.startswith('D') else res_name,  # Remove D prefix for DNA
                    'full_name': res_name,
                    'index': res_seq,
                    'chain_idx': chain_idx,
                    'residue_obj': residue,
                })
                residue_indices.append(res_seq)
                chain_indices.append(chain_idx)
    
    if not residues:
        raise ValueError(f"No RNA residues found in {cif_path}")
    
    num_res = len(residues)
    
    # Initialize atom arrays (37 atoms to match protein atom37 format)
    atom_positions = np.zeros((num_res, 37, 3), dtype=np.float32)
    atom_mask = np.zeros((num_res, 37), dtype=np.float32)
    
    # Extract atom coordinates for each residue
    for res_idx, res_info in enumerate(residues):
        res_name = res_info['name']
        residue = res_info['residue_obj']
        
        # Get expected atom names for this residue type
        expected_atoms = RNA_ATOM_NAMES.get(res_name, [])
        
        # Map atom names to indices
        atom_name_to_idx = {name: i for i, name in enumerate(expected_atoms)}
        
        # Extract atoms from residue
        for atom in residue:
            atom_name = atom.get_name()
            
            # Normalize atom name (handle quoted names from mmCIF)
            normalized_name = atom_name.strip('"')
            
            # Map to internal naming convention
            if normalized_name in ATOM_NAME_MAP:
                internal_name = ATOM_NAME_MAP[normalized_name]
            else:
                # Skip hydrogen atoms and unknown atoms
                if normalized_name.startswith('H'):
                    continue
                internal_name = normalized_name
            
            # Find position in expected atom list
            if internal_name in atom_name_to_idx:
                atom_idx = atom_name_to_idx[internal_name]
                if atom_idx < 37:  # Ensure within bounds
                    atom_positions[res_idx, atom_idx] = atom.get_coord()
                    atom_mask[res_idx, atom_idx] = 1.0
    
    # Create sequence list
    sequence = [r['name'] for r in residues]
    
    # Normalize residue indices to start from 1
    if residue_indices:
        min_idx = min(residue_indices)
        residue_indices = [idx - min_idx + 1 for idx in residue_indices]
    
    data_dict = {
        'atom_positions': atom_positions,
        'atom_mask': atom_mask,
        'residue_index': np.array(residue_indices, dtype=np.int32),
        'chain_index': np.array(chain_indices, dtype=np.int32),
    }
    
    return sequence, data_dict


def create_aatype_onehot(sequence: List[str]) -> np.ndarray:
    """Create aatype one-hot encoding for RNA.
    
    Uses 4-dim one-hot encoding for RNA nucleotides (A=0, C=1, G=2, U=3).
    This is compatible with OpenFold transforms when treated as protein UNK.
    
    The data loader will use is_rna_residue() to identify RNA residues and
    handle them appropriately.
    
    Args:
        sequence: List of nucleotide one-letter codes
        
    Returns:
        [num_res, 21] one-hot array where RNA is encoded at positions 0-3
        (can be treated as protein residues 0-3 for OpenFold compatibility)
    """
    num_res = len(sequence)
    aatype = np.zeros((num_res, 21), dtype=np.int32)  # 21 = 20 amino acids + UNK
    
    for i, nt in enumerate(sequence):
        restype_idx = RNA_RESTYPE_MAP.get(nt, 20)  # Default to UNK (20) if unknown
        # Store RNA at indices 0-3 (these will be identified by is_rna flag or sequence context)
        if 0 <= restype_idx <= 3:  # RNA types
            aatype[i, restype_idx] = 1
        else:
            aatype[i, 20] = 1  # Unknown
    
    return aatype


def create_rna_restype_array(sequence: List[str]) -> np.ndarray:
    """Create Protenix-compatible restype array for RNA.
    
    This creates the actual restype indices (21-25) that will be used
    by the model for RNA-specific processing.
    
    Args:
        sequence: List of nucleotide one-letter codes
        
    Returns:
        [num_res] array with restype indices (21-25 for RNA)
    """
    num_res = len(sequence)
    restype = np.zeros(num_res, dtype=np.int32)
    
    for i, nt in enumerate(sequence):
        restype[i] = RNA_RESTYPE_MAP.get(nt, 20)  # Direct Protenix encoding
    
    return restype


def create_trajectory_data(
    sequence: List[str],
    atom_data: Dict[str, np.ndarray],
    n_frames: int = 100,
    noise_scale: float = 0.1
) -> Dict[str, np.ndarray]:
    """Create synthetic trajectory data from single structure.
    
    In production, this would be replaced with actual MD trajectory data.
    For now, we create synthetic frames by adding small noise to coordinates.
    
    Args:
        sequence: List of nucleotide one-letter codes
        atom_data: Dictionary with atom positions, masks, etc.
        n_frames: Number of trajectory frames to generate
        noise_scale: Scale of Gaussian noise to add
        
    Returns:
        Dictionary with trajectory data compatible with DyneTrion
    """
    num_res = len(sequence)
    
    # Create aatype one-hot encoding
    aatype = create_aatype_onehot(sequence)
    
    # Create trajectory frames [n_frames, num_res, 37, 3]
    all_atom_positions = np.zeros((n_frames, num_res, 37, 3), dtype=np.float32)
    
    base_positions = atom_data['atom_positions']
    
    for frame in range(n_frames):
        if frame == 0:
            # First frame is the original structure
            all_atom_positions[frame] = base_positions
        else:
            # Add small noise for other frames (simulating dynamics)
            noise = np.random.randn(num_res, 37, 3) * noise_scale
            all_atom_positions[frame] = base_positions + noise
    
    # Apply atom mask to zero out missing atoms
    atom_mask = atom_data['atom_mask']
    for frame in range(n_frames):
        all_atom_positions[frame] *= atom_mask[..., None]
    
    return {
        'aatype': aatype,
        'all_atom_positions': all_atom_positions,
        'all_atom_mask': atom_mask,
        'residue_index': atom_data['residue_index'],
        'chain_index': atom_data['chain_index'],
        'sequence': ''.join(sequence),
    }


def validate_embedding_compatibility(embed_path: str, num_residues: int) -> bool:
    """Validate that Protenix embedding matches structure.
    
    Args:
        embed_path: Path to Protenix embedding .npz file
        num_residues: Number of residues in structure
        
    Returns:
        True if compatible, False otherwise
    """
    if not os.path.exists(embed_path):
        print(f"Warning: Embedding file not found: {embed_path}")
        return False
    
    try:
        embed_data = np.load(embed_path)
        
        # Check single_s shape (should be [num_res, 384])
        if 'single_s' in embed_data:
            single_s = embed_data['single_s']
            if single_s.shape[0] != num_residues:
                print(f"Warning: Embedding residue count mismatch. "
                      f"Expected {num_residues}, got {single_s.shape[0]}")
                return False
        
        # Check pair_z shape (should be [num_res, num_res, 128])
        if 'pair_z' in embed_data:
            pair_z = embed_data['pair_z']
            if pair_z.shape[0] != num_residues or pair_z.shape[1] != num_residues:
                print(f"Warning: Embedding pair shape mismatch. "
                      f"Expected ({num_residues}, {num_residues}), "
                      f"got ({pair_z.shape[0]}, {pair_z.shape[1]})")
                return False
        
        print(f"Embedding validated: {single_s.shape[0]} residues, "
              f"node dim={single_s.shape[1]}, edge dim={pair_z.shape[2]}")
        return True
        
    except Exception as e:
        print(f"Warning: Failed to validate embedding: {e}")
        return False


def prepare_rna_training_data(
    pdb_id: str,
    cif_path: Optional[str],
    embed_path: str,
    output_dir: str,
    n_frames: int = 100,
    download_if_missing: bool = True
) -> Tuple[str, str]:
    """Prepare all data needed for RNA training.
    
    Args:
        pdb_id: PDB ID (e.g., '2KMJ')
        cif_path: Path to mmCIF file (if None, will download)
        embed_path: Path to Protenix embedding .npz file
        output_dir: Directory to save output files
        n_frames: Number of trajectory frames to generate
        download_if_missing: Whether to download CIF if not found
        
    Returns:
        Tuple of (csv_path, npz_path)
    """
    print(f"\n{'='*60}")
    print(f"Preparing RNA training data for {pdb_id}")
    print(f"{'='*60}\n")
    
    # Create output directory
    pdb_output_dir = os.path.join(output_dir, pdb_id)
    os.makedirs(pdb_output_dir, exist_ok=True)
    
    # Download CIF if needed
    if cif_path is None or not os.path.exists(cif_path):
        if download_if_missing:
            cif_path = os.path.join(pdb_output_dir, f'{pdb_id}.cif')
            download_cif(pdb_id, cif_path)
        else:
            raise FileNotFoundError(f"CIF file not found: {cif_path}")
    
    # Parse RNA structure
    print(f"\nParsing RNA structure from {cif_path}...")
    sequence, atom_data = parse_rna_cif(cif_path)
    print(f"  Found {len(sequence)} RNA residues")
    print(f"  Sequence: {''.join(sequence)}")
    print(f"  Atoms per residue: {int(atom_data['atom_mask'].sum(axis=1).mean()):.0f} (avg)")
    
    # Validate embedding compatibility
    print(f"\nValidating Protenix embedding...")
    if not validate_embedding_compatibility(embed_path, len(sequence)):
        print(f"  Warning: Embedding validation failed. Data may be incompatible.")
    
    # Create trajectory data
    print(f"\nCreating trajectory data ({n_frames} frames)...")
    traj_data = create_trajectory_data(sequence, atom_data, n_frames=n_frames)
    
    # Create Protenix-compatible restype array
    restype_protenix = create_rna_restype_array(sequence)
    
    # Create is_rna flag array (all True for pure RNA structures)
    is_rna = np.ones(len(sequence), dtype=bool)
    
    # Save trajectory NPZ
    pos_path = os.path.join(pdb_output_dir, f'{pdb_id}.npz')
    np.savez(pos_path,
        aatype=traj_data['aatype'],
        restype_protenix=restype_protenix,  # Store Protenix-compatible encoding (22-25)
        is_rna=is_rna,  # Explicit RNA flag for data loader
        all_atom_positions=traj_data['all_atom_positions'],
        all_atom_mask=traj_data['all_atom_mask'],
        residue_index=traj_data['residue_index'],
        chain_index=traj_data['chain_index'],
        seq_length=np.array([len(sequence)], dtype=np.int32),
        sequence=np.array([traj_data['sequence']], dtype=object),
        between_segment_residues=np.zeros(len(sequence), dtype=np.int32),
        domain_name=np.array([pdb_id], dtype=object),
        resolution=np.array([1.0], dtype=np.float32),
        is_distillation=np.array(0.0, dtype=np.float32),
    )
    print(f"  Saved trajectory: {pos_path}")
    
    # Create metadata CSV
    n_res = len(sequence)
    csv_data = {
        'accession': pdb_id,
        'pdb_id': pdb_id,
        'FRAMESTEP': 1.0,
        'mdFrames': n_frames,
        'seq_num': 1,
        'total_seq_len': n_res,
        'seq_len': str([n_res]),
        'sequence': str([traj_data['sequence']]),
        'json_path': pos_path,
        'pdb_path': cif_path,
        'traj_path': pos_path,
        'embed_path': embed_path,
        'pos_path': pos_path,
    }
    
    csv_path = os.path.join(pdb_output_dir, 'metadata.csv')
    pd.DataFrame([csv_data]).to_csv(csv_path, index=False)
    print(f"  Saved metadata: {csv_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary for {pdb_id}:")
    print(f"  Residues: {n_res}")
    print(f"  Sequence: {traj_data['sequence']}")
    print(f"  Frames: {n_frames}")
    print(f"  Atom positions shape: {traj_data['all_atom_positions'].shape}")
    print(f"  Aatype shape: {traj_data['aatype'].shape}")
    print(f"  Restype encoding: A=0, C=1, G=2, U=3 (stored) -> 22-25 (Protenix)")
    print(f"{'='*60}\n")
    
    return csv_path, pos_path


def main():
    parser = argparse.ArgumentParser(
        description='Prepare RNA training data for DyneTrion from mmCIF and Protenix embeddings'
    )
    parser.add_argument('--pdb_id', type=str, required=True,
                        help='PDB ID (e.g., 2KMJ)')
    parser.add_argument('--cif_path', type=str, default=None,
                        help='Path to mmCIF file (will download if not provided)')
    parser.add_argument('--embed_path', type=str, default=None,
                        help='Path to Protenix embedding .npz file')
    parser.add_argument('--output_dir', type=str, default='rna_data',
                        help='Output directory for processed data')
    parser.add_argument('--n_frames', type=int, default=100,
                        help='Number of trajectory frames to generate')
    parser.add_argument('--protenix_embed_dir', type=str,
                        default='/inspire/ssd/project/sais-bio/public/xiangwenkai/Protenix_v1/extract_embedding/output/embeddings_npz',
                        help='Base directory for Protenix embeddings')
    
    args = parser.parse_args()
    
    # Auto-determine embed_path if not provided
    if args.embed_path is None:
        args.embed_path = os.path.join(
            args.protenix_embed_dir,
            f'{args.pdb_id}_seed101.npz'
        )
    
    # Check if embedding exists
    if not os.path.exists(args.embed_path):
        print(f"Error: Protenix embedding not found: {args.embed_path}")
        print(f"Please provide --embed_path or check --protenix_embed_dir")
        sys.exit(1)
    
    # Prepare data
    try:
        csv_path, pos_path = prepare_rna_training_data(
            pdb_id=args.pdb_id,
            cif_path=args.cif_path,
            embed_path=args.embed_path,
            output_dir=args.output_dir,
            n_frames=args.n_frames,
        )
        
        print(f"Data preparation complete!")
        print(f"  CSV: {csv_path}")
        print(f"  NPZ: {pos_path}")
        
        # Test loading the data
        print(f"\nTesting data loading...")
        test_data = np.load(pos_path)
        print(f"  Loaded NPZ keys: {list(test_data.keys())}")
        print(f"  all_atom_positions shape: {test_data['all_atom_positions'].shape}")
        print(f"  aatype shape: {test_data['aatype'].shape}")
        
        # Verify restype encoding
        aatype = test_data['aatype']
        restype_indices = np.argmax(aatype, axis=1)
        rna_mask = (restype_indices >= 21) & (restype_indices <= 24)
        num_rna = rna_mask.sum()
        print(f"  RNA residues detected: {num_rna}/{len(aatype)}")
        
        print(f"\nReady for training! Use CSV: {csv_path}")
        
    except Exception as e:
        print(f"Error preparing data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
