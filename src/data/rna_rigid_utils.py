"""RNA-specific rigid body utilities (ported from RhoFold).

This module provides RNA-specific rigid body transformation utilities
that are needed for RNA structure prediction in hybrid protein-RNA systems.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from openfold.utils.rigid_utils import Rigid, Rotation


def calc_rot_tsl_rna(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    eps: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate rotation matrix and translation vector from 3 points.
    
    Ported from RhoFold for RNA backbone frame construction.
    
    The frame is constructed such that:
    - x2 is the origin
    - x2->x3 is the x-axis
    - perpendicular to x1-x2-x3 plane is the z-axis
    
    Args:
        x1: [*, 3] coordinate tensor
        x2: [*, 3] coordinate tensor (origin)
        x3: [*, 3] coordinate tensor (x-axis direction)
        eps: Small epsilon for numerical stability
        
    Returns:
        rot_mat: [*, 3, 3] rotation matrix
        tsl_vec: [*, 3] translation vector
    """
    v1 = x3 - x2
    v2 = x1 - x2
    
    e1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)
    
    # Project out e1 component from v2
    dot = (e1 * v2).sum(dim=-1, keepdim=True)
    u2 = v2 - dot * e1
    e2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)
    
    # Complete orthonormal basis
    e3 = torch.cross(e1, e2, dim=-1)
    
    # Stack to form rotation matrix [*, 3, 3]
    rot_mat = torch.stack([e1, e2, e3], dim=-1)
    tsl_vec = x2
    
    return rot_mat, tsl_vec


def merge_rot_tsl(
    rot_1: torch.Tensor,
    tsl_1: torch.Tensor,
    rot_2: torch.Tensor,
    tsl_2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge rotation and translation transformations.
    
    Ported from RhoFold. Composes two rigid transformations:
    T_total = T_1 ∘ T_2
    
    Args:
        rot_1: [*, 3, 3] first rotation
        tsl_1: [*, 3] first translation
        rot_2: [*, 3, 3] second rotation
        tsl_2: [*, 3] second translation
        
    Returns:
        rot: [*, 3, 3] composed rotation
        tsl: [*, 3] composed translation
    """
    # rot = rot_1 @ rot_2
    rot = torch.matmul(rot_1, rot_2)
    # tsl = rot_1 @ tsl_2 + tsl_1
    tsl = torch.matmul(rot_1, tsl_2.unsqueeze(-1)).squeeze(-1) + tsl_1
    return rot, tsl


def calc_angl_rot_tsl(
    angl: torch.Tensor,
    eps: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate rotation and translation from angle tensor [cos, sin].
    
    Ported from RhoFold for RNA torsion angles.
    
    Args:
        angl: [*, 2] tensor with [cos(angle), sin(angle)]
        eps: Small epsilon for numerical stability
        
    Returns:
        rot: [*, 3, 3] rotation matrix around x-axis
        tsl: [*, 3] zero translation
    """
    # Extract cos and sin
    cos_vec = angl[..., 0]  # [*, 1]
    sin_vec = angl[..., 1]  # [*, 1]
    
    # Create rotation matrix around x-axis
    # [[1, 0, 0],
    #  [0, cos, -sin],
    #  [0, sin, cos]]
    
    zeros = torch.zeros_like(cos_vec)
    ones = torch.ones_like(cos_vec)
    
    rot = torch.stack([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, cos_vec, -sin_vec], dim=-1),
        torch.stack([zeros, sin_vec, cos_vec], dim=-1),
    ], dim=-2)  # [*, 3, 3]
    
    tsl = torch.zeros_like(angl[..., :1]).expand(*angl.shape[:-1], 3)
    
    return rot, tsl


def build_rna_backbone_frame(
    atom_positions: torch.Tensor,
    atom_mask: Optional[torch.Tensor] = None
) -> Rigid:
    """Build RNA backbone frames from atom positions.
    
    RNA backbone frame is defined using C4' as origin, with:
    - C4' at origin (analogous to CA in protein)
    - C4'->P direction as x-axis (analogous to C in protein)
    - C4'->C1' direction for y-plane (analogous to N in protein)
    
    Args:
        atom_positions: [..., N, 23, 3] RNA atom positions in RhoFold ordering
        atom_mask: [..., N, 23] optional atom mask
        
    Returns:
        rigids: [..., N] Rigid objects for RNA backbone
    """
    # RNA atom indices in 23-atom representation
    C4_PRIME_IDX = 0   # C4' - sugar ring atom (origin)
    C1_PRIME_IDX = 1   # C1' - anomeric carbon (y-plane)
    P_IDX = 19         # P - phosphate (x-axis, note: this is index 19 not 18)
    
    # Get frame-defining atoms
    c4_pos = atom_positions[..., C4_PRIME_IDX, :]   # [..., N, 3]
    c1_pos = atom_positions[..., C1_PRIME_IDX, :]   # [..., N, 3]
    p_pos = atom_positions[..., P_IDX, :]           # [..., N, 3]
    
    # Build frames using RNA-specific frame construction
    rot, tsl = calc_rot_tsl_rna(c1_pos, c4_pos, p_pos)
    
    # Create Rigid object
    rigids = Rigid(Rotation(rot_mats=rot), tsl)
    
    if atom_mask is not None:
        # Check that all 3 atoms are present
        valid_mask = (atom_mask[..., C4_PRIME_IDX] * 
                     atom_mask[..., C1_PRIME_IDX] * 
                     atom_mask[..., P_IDX])
        # Zero out invalid frames
        rigids = rigids * valid_mask[..., None]
    
    return rigids


def build_hybrid_backbone_frames(
    atom_positions: torch.Tensor,
    aatype: torch.Tensor,
    atom_mask: Optional[torch.Tensor] = None,
    is_rna_mask: Optional[torch.Tensor] = None
) -> Rigid:
    """Build backbone frames for hybrid protein-RNA structures.
    
    Uses different frame conventions for protein vs RNA:
    - Protein: N-CA-C frame (CA at origin)
    - RNA: C4'-C1'-P frame (C4' at origin)
    
    Args:
        atom_positions: [..., N, num_atoms, 3] atom positions
                       For protein: 37 atoms, for RNA: 23 atoms
        aatype: [..., N] residue type indices (0-25, RNA at 22-25)
        atom_mask: [..., N, num_atoms] atom mask
        is_rna_mask: [..., N] optional pre-computed RNA mask
        
    Returns:
        rigids: [..., N] Rigid objects for all residues
    """
    # Detect RNA residues if mask not provided
    # Use vectorized operations to avoid GPU->CPU->GPU transfer
    if is_rna_mask is None:
        is_rna_mask = (aatype >= 22) & (aatype <= 25)
    
    is_protein_mask = ~is_rna_mask
    
    # Initialize output
    batch_shape = aatype.shape
    device = aatype.device
    
    # Get backbone atoms for proteins (N, CA, C)
    # These are standard PDB atom names
    N_IDX = 0   # N atom index in atom37
    CA_IDX = 1  # CA atom index in atom37  
    C_IDX = 2   # C atom index in atom37
    
    # Initialize identity frames
    all_rot = torch.eye(3, device=device).expand(*batch_shape, 3, 3).clone()
    all_trans = torch.zeros(*batch_shape, 3, device=device)
    
    # Process protein residues
    if is_protein_mask.any():
        n_pos = atom_positions[..., N_IDX, :]
        ca_pos = atom_positions[..., CA_IDX, :]
        c_pos = atom_positions[..., C_IDX, :]
        
        # Use OpenFold's protein frame construction
        prot_rot, prot_trans = calc_rot_tsl_rna(n_pos, ca_pos, c_pos)
        
        all_rot = torch.where(
            is_protein_mask[..., None, None],
            prot_rot,
            all_rot
        )
        all_trans = torch.where(
            is_protein_mask[..., None],
            prot_trans,
            all_trans
        )
    
    # Process RNA residues
    if is_rna_mask.any():
        # RNA uses C4', C1', P for frame construction
        C4_PRIME_IDX = 0
        C1_PRIME_IDX = 1
        P_IDX = 19
        
        c4_pos = atom_positions[..., C4_PRIME_IDX, :]
        c1_pos = atom_positions[..., C1_PRIME_IDX, :]
        p_pos = atom_positions[..., P_IDX, :]
        
        rna_rot, rna_trans = calc_rot_tsl_rna(c1_pos, c4_pos, p_pos)
        
        all_rot = torch.where(
            is_rna_mask[..., None, None],
            rna_rot,
            all_rot
        )
        all_trans = torch.where(
            is_rna_mask[..., None],
            rna_trans,
            all_trans
        )
    
    rigids = Rigid(Rotation(rot_mats=all_rot), all_trans)
    
    if atom_mask is not None:
        # Apply mask based on backbone atoms
        # For protein: check N (0), CA (1), C (2)
        # For RNA: check C4' (0), C1' (1), P (19)
        protein_bb_mask = (atom_mask[..., 0] * atom_mask[..., 1] * atom_mask[..., 2])
        rna_bb_mask = (atom_mask[..., 0] * atom_mask[..., 1] * atom_mask[..., 19])
        valid_mask = torch.where(
            is_protein_mask,
            protein_bb_mask,
            rna_bb_mask
        )
        rigids = rigids * valid_mask[..., None]
    
    return rigids


class RNAFrameBuilder:
    """Builds RNA frames and coordinates from torsion angles.
    
    Adapted from RhoFold's RNAConverter for integration with DyneTrion.
    """
    
    def __init__(self):
        """Initialize RNA frame builder with constants."""
        self._init_rna_constants()
    
    def _init_rna_constants(self):
        """Initialize RNA constants from RhoFold."""
        try:
            from rhofold.utils.constants import RNA_CONSTANTS
            self.rna_constants = RNA_CONSTANTS
            self.has_rhofold = True
        except ImportError:
            # Fallback: define minimal constants
            self.rna_constants = None
            self.has_rhofold = False
            
    def build_cords_from_frames(
        self,
        seq: str,
        frames: torch.Tensor,  # [N, 7] quat(4) + trans(3)
        angles: torch.Tensor,  # [N, 4, 2] sin/cos for 4 angles
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build RNA coordinates from frames and angles.
        
        This is a simplified version that uses RhoFold's converter.
        For full implementation, see rhofold/utils/converter.py
        
        Args:
            seq: RNA sequence string ('A', 'G', 'U', 'C')
            frames: [N, 7] backbone frames as quaternion + translation
            angles: [N, 4, 2] torsion angles
            
        Returns:
            coords: [N, 23, 3] atom coordinates
            mask: [N, 23] atom mask
        """
        if self.has_rhofold:
            from rhofold.utils.converter import RNAConverter
            from rhofold.utils.rigid_utils import Rigid as RhoRigid
            
            converter = RNAConverter()
            
            # Convert frames to RhoFold's Rigid format
            rigid = RhoRigid.from_tensor_7(frames, normalize_quats=True)
            
            # Build coordinates
            cord, cmsk = converter.build_cords(seq, frames, angles, rtn_cmsk=True)
            
            return cord, cmsk
        else:
            # Fallback: return frames as pseudo-coordinates
            N = len(seq)
            coords = torch.zeros(N, 23, 3, device=frames.device)
            mask = torch.zeros(N, 23, device=frames.device)
            # Set backbone atoms to frame positions
            coords[:, 0, :] = frames[:, 4:7]  # C4' at frame translation
            mask[:, 0] = 1.0
            return coords, mask
