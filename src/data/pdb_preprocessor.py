"""PDB preprocessing for end-to-end structure prediction training."""

import os
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from Bio.PDB import PDBParser

from src.data import protein
from src.data import all_atom
from openfold.utils import rigid_utils as ru
from openfold.np import residue_constants as rc


def compute_dihedral(p0, p1, p2, p3):
    """Compute dihedral angle from 4 points."""
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1)
    if b1_norm < 1e-7:
        return None
    b1 = b1 / b1_norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)


def extract_torsion_angles_from_pdb(
    pdb_path: str,
    chain_id: str,
    aatype: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract torsion angles from a PDB file for a specific chain.
    
    Args:
        pdb_path: Path to PDB file
        chain_id: Chain ID to extract angles from
        aatype: Array of aatype indices for residues in the chain
        
    Returns:
        torsion_gt: [N, 7, 2] sin/cos of torsion angles
        torsion_mask: [N, 7] mask of valid angles
        alt_torsion_gt: [N, 7, 2] alternate angles for pi-periodic chi
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    model = list(structure.get_models())[0]
    chain = model[chain_id]
    residues = list(chain.get_residues())
    
    N = len(aatype)
    torsion_gt = np.zeros((N, 7, 2), dtype=np.float32)
    torsion_mask = np.zeros((N, 7), dtype=np.float32)
    
    # Map PDB residues to aatype indices by matching residue names
    residue_map = {}
    for i, res in enumerate(residues):
        # Use residue sequence number
        res_idx = res.id[1] - 1  # Convert to 0-indexed
        if 0 <= res_idx < N:
            residue_map[res_idx] = res
    
    for res_idx in range(N):
        if res_idx not in residue_map:
            continue
        res = residue_map[res_idx]
        
        # phi: C(i-1) - N(i) - CA(i) - C(i)
        if res_idx > 0 and (res_idx - 1) in residue_map:
            prev_res = residue_map[res_idx - 1]
            if 'C' in prev_res and 'N' in res and 'CA' in res and 'C' in res:
                angle = compute_dihedral(
                    prev_res['C'].get_coord(),
                    res['N'].get_coord(),
                    res['CA'].get_coord(),
                    res['C'].get_coord(),
                )
                if angle is not None:
                    torsion_gt[res_idx, 0] = [np.cos(angle), np.sin(angle)]
                    torsion_mask[res_idx, 0] = 1.0
        
        # psi: N(i) - CA(i) - C(i) - N(i+1)
        if res_idx < N - 1 and (res_idx + 1) in residue_map:
            next_res = residue_map[res_idx + 1]
            if 'N' in res and 'CA' in res and 'C' in res and 'N' in next_res:
                angle = compute_dihedral(
                    res['N'].get_coord(),
                    res['CA'].get_coord(),
                    res['C'].get_coord(),
                    next_res['N'].get_coord(),
                )
                if angle is not None:
                    torsion_gt[res_idx, 1] = [np.cos(angle), np.sin(angle)]
                    torsion_mask[res_idx, 1] = 1.0
        
        # omega: CA(i) - C(i) - N(i+1) - CA(i+1)
        if res_idx < N - 1 and (res_idx + 1) in residue_map:
            next_res = residue_map[res_idx + 1]
            if 'CA' in res and 'C' in res and 'N' in next_res and 'CA' in next_res:
                angle = compute_dihedral(
                    res['CA'].get_coord(),
                    res['C'].get_coord(),
                    next_res['N'].get_coord(),
                    next_res['CA'].get_coord(),
                )
                if angle is not None:
                    torsion_gt[res_idx, 2] = [np.cos(angle), np.sin(angle)]
                    torsion_mask[res_idx, 2] = 1.0
        
        # chi angles
        chi_atoms = rc.chi_angles_atoms.get(res.resname, [])
        for chi_idx, atom_names in enumerate(chi_atoms):
            if all(an in res for an in atom_names):
                coords = [res[an].get_coord() for an in atom_names]
                angle = compute_dihedral(*coords)
                if angle is not None:
                    torsion_gt[res_idx, 3 + chi_idx] = [np.cos(angle), np.sin(angle)]
                    torsion_mask[res_idx, 3 + chi_idx] = 1.0
    
    # Compute alternate torsion angles for pi-periodic chi angles
    alt_torsion_gt = torsion_gt.copy()
    for res_type in range(20):
        pi_periodic = rc.chi_pi_periodic[res_type]
        mask_res = (aatype == res_type)
        for chi_idx in range(4):
            if pi_periodic[chi_idx]:
                alt_torsion_gt[mask_res, 3 + chi_idx, 0] *= -1
                alt_torsion_gt[mask_res, 3 + chi_idx, 1] *= -1
    
    return torsion_gt, torsion_mask, alt_torsion_gt


def preprocess_pdb(
    pdb_path: str,
    chain_id: Optional[str] = None,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, torch.Tensor]:
    """Preprocess a PDB file into features for end-to-end training.
    
    Args:
        pdb_path: Path to PDB file
        chain_id: Chain ID to parse. If None, parses all chains.
        device: Torch device for output tensors
        
    Returns:
        Dictionary with:
            - aatype: [N] amino acid type indices
            - atom37_pos: [N, 37, 3] atom positions
            - atom37_mask: [N, 37] atom mask
            - rigids_0: [N, 7] backbone rigids (quat + trans)
            - torsion_angles_sin_cos: [N, 7, 2]
            - alt_torsion_angles_sin_cos: [N, 7, 2]
            - torsion_angles_mask: [N, 7]
            - seq_idx: [N] sequence indices
            - residue_index: [N] PDB residue indices
    """
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    
    # Parse PDB - handle multi-model structures by using first model
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    # Extract first model only
    models = list(structure.get_models())
    if len(models) > 1:
        # Reconstruct PDB string with only first model
        pdb_lines = []
        in_first_model = False
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('MODEL'):
                    if not in_first_model:
                        in_first_model = True
                        pdb_lines.append(line)
                    else:
                        break
                elif line.startswith('ENDMDL'):
                    pdb_lines.append(line)
                    break
                elif in_first_model or line.startswith(('ATOM', 'HETATM', 'TER')):
                    pdb_lines.append(line)
        
        if not in_first_model:
            # No MODEL records, just use all ATOM lines
            pdb_lines = [l for l in open(pdb_path, 'r') if l.startswith(('ATOM', 'HETATM', 'TER', 'END'))]
        
        pdb_str = ''.join(pdb_lines)
    else:
        with open(pdb_path, 'r') as f:
            pdb_str = f.read()
    
    prot = protein.from_pdb_string(pdb_str, chain_id=chain_id)
    
    atom37_pos = prot.atom_positions  # [N, 37, 3]
    atom37_mask = prot.atom_mask  # [N, 37]
    aatype = prot.aatype  # [N]
    residue_index = prot.residue_index  # [N]
    
    N = len(aatype)
    
    # Build backbone rigids from N, CA, C
    atom37_torch = torch.from_numpy(atom37_pos).float().to(device)
    atom37_mask_torch = torch.from_numpy(atom37_mask).float().to(device)
    aatype_torch = torch.from_numpy(aatype).long().to(device)
    
    n_pos = atom37_torch[:, 0]
    ca_pos = atom37_torch[:, 1]
    c_pos = atom37_torch[:, 2]
    
    bb_rigids = ru.Rigid.from_3_points(n_pos, ca_pos, c_pos)
    rigids_0 = bb_rigids.to_tensor_7()  # [N, 7]
    
    # Compute torsion angles using OpenFold's data_transforms
    from openfold.data import data_transforms
    prot_feats = {
        'aatype': aatype_torch.unsqueeze(0),  # [1, N]
        'all_atom_positions': atom37_torch.unsqueeze(0),  # [1, N, 37, 3]
        'all_atom_mask': atom37_mask_torch.unsqueeze(0),  # [1, N, 37]
    }
    torsion_angles_feats = data_transforms.atom37_to_torsion_angles()(prot_feats)
    
    torsion_angles = torsion_angles_feats['torsion_angles_sin_cos'][0].cpu().numpy()  # [N, 7, 2]
    alt_torsion_angles = torsion_angles_feats['alt_torsion_angles_sin_cos'][0].cpu().numpy()  # [N, 7, 2]
    torsion_angles_mask = torsion_angles_feats['torsion_angles_mask'][0].cpu().numpy()  # [N, 7]
    
    # Also try extracting more precise torsion angles from raw PDB
    # If chain_id is not specified, use the first chain
    if chain_id is None:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        first_chain = list(structure.get_models())[0].get_chains().__next__()
        chain_id = first_chain.id
    
    try:
        pdb_torsion_gt, pdb_torsion_mask, pdb_alt_torsion = extract_torsion_angles_from_pdb(
            pdb_path, chain_id, aatype
        )
        # Use PDB-extracted angles where available (more accurate for side chains)
        # but keep OpenFold's backbone angles which are more robust
        mask_3d = pdb_torsion_mask[..., None]
        torsion_angles = np.where(mask_3d, pdb_torsion_gt, torsion_angles)
        alt_torsion_angles = np.where(mask_3d, pdb_alt_torsion, alt_torsion_angles)
        torsion_angles_mask = np.maximum(torsion_angles_mask, pdb_torsion_mask)
    except Exception as e:
        print(f"Warning: Could not extract precise torsion angles from PDB: {e}")
    
    return {
        'aatype': aatype_torch,
        'atom37_pos': atom37_torch,
        'atom37_mask': atom37_mask_torch,
        'rigids_0': rigids_0,
        'torsion_angles_sin_cos': torch.from_numpy(torsion_angles).float().to(device),
        'alt_torsion_angles_sin_cos': torch.from_numpy(alt_torsion_angles).float().to(device),
        'torsion_angles_mask': torch.from_numpy(torsion_angles_mask).float().to(device),
        'seq_idx': torch.arange(N, device=device).unsqueeze(0),  # [1, N]
        'residue_index': torch.from_numpy(residue_index).long().to(device).unsqueeze(0),
    }


def preprocess_pdb_for_training(
    pdb_path: str,
    chain_id: Optional[str] = None,
    scale_factor: float = 1.0,
    center: bool = True,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, torch.Tensor]:
    """Preprocess PDB with optional centering and scaling for training.
    
    Args:
        pdb_path: Path to PDB file
        chain_id: Chain ID to parse
        scale_factor: Scale coordinates by this factor
        center: Whether to center coordinates at origin
        device: Torch device
        
    Returns:
        Preprocessed features ready for training
    """
    features = preprocess_pdb(pdb_path, chain_id=chain_id, device=device)
    
    if center or scale_factor != 1.0:
        atom37_pos = features['atom37_pos']
        ca_idx = 1
        ca_pos = atom37_pos[:, ca_idx]
        ca_mask = features['atom37_mask'][:, ca_idx]
        
        if center:
            bb_center = ca_pos.sum(dim=0) / (ca_mask.sum() + 1e-5)
            atom37_pos = atom37_pos - bb_center[None, None, :]
        
        if scale_factor != 1.0:
            atom37_pos = atom37_pos / scale_factor
        
        features['atom37_pos'] = atom37_pos
        
        # Recompute rigids after centering/scaling
        n_pos = atom37_pos[:, 0]
        ca_pos = atom37_pos[:, 1]
        c_pos = atom37_pos[:, 2]
        bb_rigids = ru.Rigid.from_3_points(n_pos, ca_pos, c_pos)
        features['rigids_0'] = bb_rigids.to_tensor_7()
    
    return features
