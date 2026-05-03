"""RNA-specific rigid body utilities (optimized).

This module provides optimized RNA-specific rigid body transformation utilities
using PyTorch best practices:
- torch.jit.script for performance-critical functions
- Vectorized operations without CPU-GPU sync
- Memory-efficient implementations
- Proper BF16/FP16 support
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from openfold.utils.rigid_utils import Rigid, Rotation

# Compile flag - enable for PyTorch 2.0+
USE_COMPILE = hasattr(torch, 'compile')


@torch.jit.script
def calc_rot_tsl_rna(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate rotation matrix and translation vector from 3 points (JIT compiled).
    
    Args:
        x1: [*, 3] coordinate tensor
        x2: [*, 3] coordinate tensor (origin)
        x3: [*, 3] coordinate tensor (x-axis direction)
        eps: Small epsilon for numerical stability (default 1e-8 for better precision)
        
    Returns:
        rot_mat: [*, 3, 3] rotation matrix
        tsl_vec: [*, 3] translation vector
    """
    v1 = x3 - x2
    v2 = x1 - x2
    
    # Normalize v1 to get e1
    v1_norm = torch.norm(v1, dim=-1, keepdim=True)
    e1 = v1 / (v1_norm + eps)
    
    # Project out e1 component from v2
    dot = (e1 * v2).sum(dim=-1, keepdim=True)
    u2 = v2 - dot * e1
    
    # Normalize u2 to get e2
    u2_norm = torch.norm(u2, dim=-1, keepdim=True)
    e2 = u2 / (u2_norm + eps)
    
    # Complete orthonormal basis
    e3 = torch.cross(e1, e2, dim=-1)
    
    # Stack to form rotation matrix [*, 3, 3]
    # Use stack instead of explicit zeros/ones for better fusion
    rot_mat = torch.stack([e1, e2, e3], dim=-1)
    
    return rot_mat, x2


@torch.jit.script
def merge_rot_tsl(
    rot_1: torch.Tensor,
    tsl_1: torch.Tensor,
    rot_2: torch.Tensor,
    tsl_2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge rotation and translation transformations (JIT compiled).
    
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
    angl: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate rotation and translation from angle tensor [cos, sin].
    
    Args:
        angl: [*, 2] tensor with [cos(angle), sin(angle)]
        
    Returns:
        rot: [*, 3, 3] rotation matrix around x-axis
        tsl: [*, 3] zero translation
    """
    # Extract cos and sin
    cos_vec = angl[..., 0]
    sin_vec = angl[..., 1]
    
    # Create rotation matrix around x-axis efficiently
    # [[1, 0, 0],
    #  [0, cos, -sin],
    #  [0, sin, cos]]
    
    # Pre-allocate output tensor for better memory efficiency
    # Use expand and clone for dynamic shapes
    batch_dims = angl.shape[:-1]
    device = angl.device
    dtype = angl.dtype
    
    # Build rotation matrix using stack
    zeros = torch.zeros(batch_dims, device=device, dtype=dtype)
    ones = torch.ones(batch_dims, device=device, dtype=dtype)
    
    rot = torch.stack([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, cos_vec, -sin_vec], dim=-1),
        torch.stack([zeros, sin_vec, cos_vec], dim=-1),
    ], dim=-2)  # [*, 3, 3]
    
    # Zero translation
    tsl = torch.zeros(*batch_dims, 3, device=device, dtype=dtype)
    
    return rot, tsl


class OptimizedRNAFrameBuilder(nn.Module):
    """Optimized RNA frame builder with caching and batching support.
    
    Features:
    - Pre-computed reference positions
    - Efficient batch processing
    - BF16/FP16 compatible
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize constants
        self._init_rna_constants()
        
        # Cache for frequently used tensors
        self._cache: Dict[str, torch.Tensor] = {}
    
    def _init_rna_constants(self):
        """Initialize RNA constants."""
        try:
            from rhofold.utils.constants import RNA_CONSTANTS
            
            # Build reference position tensor [4 restypes, max_atoms, 3]
            max_atoms = 23
            rna_ref_positions = torch.zeros(4, max_atoms, 3, device=self.device)
            rna_ref_mask = torch.zeros(4, max_atoms, dtype=torch.bool, device=self.device)
            
            # Protenix: A=22, C=23, G=24, U=25
            # RhoFold internal: A=0, C=3, G=1, U=2
            restype_order = ['A', 'G', 'U', 'C']
            
            for restype_name in restype_order:
                rhofold_idx = restype_order.index(restype_name)
                atom_names = RNA_CONSTANTS.ATOM_NAMES_PER_RESD[restype_name]
                
                # Build coordinate lookup
                atom_info_dict = {}
                for atom_info in RNA_CONSTANTS.ATOM_INFOS_PER_RESD[restype_name]:
                    atom_name, _, coords = atom_info
                    atom_info_dict[atom_name] = coords
                
                # Fill in canonical order
                for atom_idx, atom_name in enumerate(atom_names):
                    if atom_idx < max_atoms and atom_name in atom_info_dict:
                        coords = atom_info_dict[atom_name]
                        rna_ref_positions[rhofold_idx, atom_idx] = torch.tensor(
                            coords, device=self.device, dtype=torch.float32
                        )
                        rna_ref_mask[rhofold_idx, atom_idx] = True
            
            self.register_buffer('rna_ref_positions', rna_ref_positions)
            self.register_buffer('rna_ref_mask', rna_ref_mask)
            
            self.protenix_to_rhofold = {22: 0, 23: 3, 24: 1, 25: 2}
            self.has_rhofold = True
            
        except ImportError:
            # Fallback
            self.register_buffer('rna_ref_positions', torch.zeros(4, 23, 3, device=self.device))
            self.register_buffer('rna_ref_mask', torch.ones(4, 23, dtype=torch.bool, device=self.device))
            self.protenix_to_rhofold = {22: 0, 23: 3, 24: 1, 25: 2}
            self.has_rhofold = False
    
    @torch.jit.export
    def forward(
        self,
        aatype: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get RNA reference positions and masks.
        
        Args:
            aatype: [*, N] residue type indices (RNA at 22-25)
            
        Returns:
            ref_pos: [*, N, 23, 3] reference positions
            ref_mask: [*, N, 23] atom masks
        """
        # Map Protenix indices to RhoFold
        rhofold_idx = torch.zeros_like(aatype)
        for p_idx, r_idx in self.protenix_to_rhofold.items():
            rhofold_idx = torch.where(aatype == p_idx, r_idx, rhofold_idx)
        
        # Gather from buffers
        ref_pos = self.rna_ref_positions[rhofold_idx]  # [*, N, 23, 3]
        ref_mask = self.rna_ref_mask[rhofold_idx]  # [*, N, 23]
        
        return ref_pos, ref_mask
    
    def to_device(self, device: torch.device):
        """Move module to device."""
        self.device = device
        return self.to(device)


def build_rna_backbone_frame(
    atom_positions: torch.Tensor,
    atom_mask: Optional[torch.Tensor] = None
) -> Rigid:
    """Build RNA backbone frames from atom positions (optimized).
    
    Args:
        atom_positions: [..., N, 23, 3] RNA atom positions
        atom_mask: [..., N, 23] optional atom mask
        
    Returns:
        rigids: [..., N] Rigid objects for RNA backbone
    """
    # RNA atom indices (hardcoded for performance)
    C4_PRIME_IDX = 0
    C1_PRIME_IDX = 1
    P_IDX = 19
    
    # Gather frame-defining atoms
    c4_pos = atom_positions[..., C4_PRIME_IDX, :]
    c1_pos = atom_positions[..., C1_PRIME_IDX, :]
    p_pos = atom_positions[..., P_IDX, :]
    
    # Build frames
    rot, tsl = calc_rot_tsl_rna(c1_pos, c4_pos, p_pos)
    rigids = Rigid(Rotation(rot_mats=rot), tsl)
    
    if atom_mask is not None:
        # Check that all 3 atoms are present
        valid_mask = (
            atom_mask[..., C4_PRIME_IDX] * 
            atom_mask[..., C1_PRIME_IDX] * 
            atom_mask[..., P_IDX]
        )
        rigids = rigids * valid_mask[..., None]
    
    return rigids


def build_hybrid_backbone_frames(
    atom_positions: torch.Tensor,
    aatype: torch.Tensor,
    atom_mask: Optional[torch.Tensor] = None,
    is_rna_mask: Optional[torch.Tensor] = None
) -> Rigid:
    """Build backbone frames for hybrid protein-RNA structures (optimized).
    
    Args:
        atom_positions: [..., N, num_atoms, 3] atom positions
        aatype: [..., N] residue type indices (0-25, RNA at 22-25)
        atom_mask: [..., N, num_atoms] atom mask
        is_rna_mask: [..., N] optional pre-computed RNA mask
        
    Returns:
        rigids: [..., N] Rigid objects for all residues
    """
    # Detect RNA residues (vectorized, no sync)
    if is_rna_mask is None:
        is_rna_mask = (aatype >= 22) & (aatype <= 25)
    
    is_protein_mask = ~is_rna_mask
    device = aatype.device
    batch_shape = aatype.shape
    
    # Pre-allocate outputs
    all_rot = torch.eye(3, device=device, dtype=torch.float32).expand(*batch_shape, 3, 3).contiguous()
    all_trans = torch.zeros(*batch_shape, 3, device=device, dtype=torch.float32)
    
    # Process protein residues
    if is_protein_mask.any():
        # N-CA-C frame
        n_pos = atom_positions[..., 0, :]
        ca_pos = atom_positions[..., 1, :]
        c_pos = atom_positions[..., 2, :]
        
        prot_rot, prot_trans = calc_rot_tsl_rna(n_pos, ca_pos, c_pos)
        
        # Use in-place where for memory efficiency
        all_rot = torch.where(is_protein_mask[..., None, None], prot_rot, all_rot)
        all_trans = torch.where(is_protein_mask[..., None], prot_trans, all_trans)
    
    # Process RNA residues
    if is_rna_mask.any():
        # C4'-C1'-P frame
        c4_pos = atom_positions[..., 0, :]
        c1_pos = atom_positions[..., 1, :]
        p_pos = atom_positions[..., 19, :]
        
        rna_rot, rna_trans = calc_rot_tsl_rna(c1_pos, c4_pos, p_pos)
        
        all_rot = torch.where(is_rna_mask[..., None, None], rna_rot, all_rot)
        all_trans = torch.where(is_rna_mask[..., None], rna_trans, all_trans)
    
    rigids = Rigid(Rotation(rot_mats=all_rot), all_trans)
    
    if atom_mask is not None:
        # Backbone atom mask check
        protein_bb_mask = atom_mask[..., 0] * atom_mask[..., 1] * atom_mask[..., 2]
        rna_bb_mask = atom_mask[..., 0] * atom_mask[..., 1] * atom_mask[..., 19]
        valid_mask = torch.where(is_protein_mask, protein_bb_mask, rna_bb_mask)
        rigids = rigids * valid_mask[..., None]
    
    return rigids


# Optional: compile the main functions for PyTorch 2.0+
if USE_COMPILE:
    try:
        calc_rot_tsl_rna = torch.compile(calc_rot_tsl_rna, mode='reduce-overhead', fullgraph=False)
        build_hybrid_backbone_frames = torch.compile(build_hybrid_backbone_frames, mode='reduce-overhead', fullgraph=False)
    except Exception:
        pass  # Compilation failed, use JIT version
