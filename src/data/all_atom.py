"""Utilities for calculating all atom representations."""
import torch
from src.data import residue_constants
from openfold.utils import rigid_utils as ru
from openfold.data import data_transforms
from openfold.utils import feats
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Rigid = ru.Rigid
Rotation = ru.Rotation

IDEALIZED_POS37 = torch.tensor(residue_constants.restype_atom37_rigid_group_positions).to(device).float()
IDEALIZED_POS37_MASK = torch.any(IDEALIZED_POS37, axis=-1).to(device)
IDEALIZED_POS = torch.tensor(residue_constants.restype_atom14_rigid_group_positions).to(device).float()
DEFAULT_FRAMES = torch.tensor(residue_constants.restype_rigid_group_default_frame).to(device).float()
ATOM_MASK = torch.tensor(residue_constants.restype_atom14_mask).to(device).float()
GROUP_IDX = torch.tensor(residue_constants.restype_atom14_to_rigid_group).to(device).long()

GROUP_IDX_37 = torch.tensor(residue_constants.restype_atom37_to_rigid_group).to(device).long()
ATOM_MASK_37 = torch.tensor(residue_constants.restype_atom37_mask).to(device).float()
IDEALIZED_POS_37 = torch.tensor(residue_constants.restype_atom37_rigid_group_positions).to(device).float()

def torsion_angles_to_frames(
        r: Rigid,
        alpha: torch.Tensor,
        aatype: torch.Tensor,
    ):
    """Conversion method of torsion angles to frames provided the backbone.
    
    Args:
        r: Backbone rigid groups.
        alpha: Torsion angles.
        aatype: residue types.
    
    Returns:
        All 8 frames corresponding to each torsion frame.

    """
    # Ensure aatype is on the same device as DEFAULT_FRAMES for indexing
    # [*, N, 8, 4, 4]
    aatype_for_index = aatype.to(device=DEFAULT_FRAMES.device).long()
    default_4x4 = DEFAULT_FRAMES[aatype_for_index, ...].to(r.device)

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat(
        [bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2
    )

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def prot_to_torsion_angles(aatype, atom37, atom37_mask):
    """Calculate torsion angle features from protein features."""
    prot_feats = {
        'aatype': aatype,
        'all_atom_positions': atom37,
        'all_atom_mask': atom37_mask,
    }
    torsion_angles_feats = data_transforms.atom37_to_torsion_angles()(prot_feats)
    torsion_angles = torsion_angles_feats['torsion_angles_sin_cos']
    torsion_mask = torsion_angles_feats['torsion_angles_mask']
    return torsion_angles, torsion_mask 


def frames_to_atom14_pos(
        r: Rigid,
        aatype: torch.Tensor,
    ):
    """Convert frames to their idealized all atom representation.

    Args:
        r: All rigid groups. [..., N, 8, 3]
        aatype: Residue types. [..., N]

    Returns:

    """

    # [*, N, 14]
    aatype = aatype.to(GROUP_IDX.device)
    group_mask = GROUP_IDX[aatype, ...]

    # [*, N, 14, 8]
    group_mask = torch.nn.functional.one_hot(
        group_mask,
        num_classes=DEFAULT_FRAMES.shape[-3],
    ).to(r.device)

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )

    # [*, N, 14, 1]
    frame_atom_mask = ATOM_MASK[aatype, ...].unsqueeze(-1).to(r.device)

    # [*, N, 14, 3]
    frame_null_pos = IDEALIZED_POS[aatype, ...].to(r.device)
    pred_positions = t_atoms_to_global.apply(frame_null_pos)
    pred_positions = pred_positions * frame_atom_mask

    return pred_positions


def compute_backbone(bb_rigids, psi_torsions):
    torsion_angles = torch.tile(
        psi_torsions[..., None, :],
        tuple([1 for _ in range(len(bb_rigids.shape))]) + (7, 1)
    ).to(bb_rigids.device)
    aatype = torch.zeros(bb_rigids.shape).long()
    # aatype = torch.zeros(bb_rigids.shape).long().to(bb_rigids.device)
    all_frames = feats.torsion_angles_to_frames(
        bb_rigids,
        torsion_angles,
        aatype,
        DEFAULT_FRAMES.to(bb_rigids.device))
    atom14_pos = frames_to_atom14_pos(
        all_frames,
        aatype)
    atom37_bb_pos = torch.zeros(bb_rigids.shape + (37, 3), device=bb_rigids.device)
    # atom14 bb order = ['N', 'CA', 'C', 'O', 'CB']
    # atom37 bb order = ['N', 'CA', 'C', 'CB', 'O']
    # TODO just leverage 'N', 'CA', 'C', 'O', 'CB' here
    atom37_bb_pos[..., :3, :] = atom14_pos[..., :3, :]
    atom37_bb_pos[..., 3, :] = atom14_pos[..., 4, :] 
    atom37_bb_pos[..., 4, :] = atom14_pos[..., 3, :]
    atom37_mask = torch.any(atom37_bb_pos, dim=-1)
    return atom37_bb_pos, atom37_mask, aatype, atom14_pos


def compute_backbone_atom37(bb_rigids,aatypes, torsions):

    torsion_angles = torsions.to(bb_rigids.device)
    aatype = aatypes.long()
    all_frames = feats.torsion_angles_to_frames(
        bb_rigids,
        torsion_angles,
        aatype,
        DEFAULT_FRAMES)# [*, N, 37]
    
    atom37_bb_pos = frames_to_atom37_pos(all_frames,aatype)

    # Use the actual atom mask from residue constants instead of checking
    # if positions are non-zero (atoms like CA/C1' can be at origin in
    # idealized coordinates).
    aatype_for_mask = aatype.to(device=ATOM_MASK_37.device).long()
    atom37_mask = ATOM_MASK_37[aatype_for_mask, ...].to(atom37_bb_pos.device)

    return atom37_bb_pos, atom37_mask, aatype, 0

def frames_to_atom37_pos(r: Rigid, aatype: torch.Tensor):
    # Ensure aatype is on the same device as constants for indexing
    aatype_device = aatype.device
    aatype_for_index = aatype.to(device=GROUP_IDX_37.device).long()
    group_idx = GROUP_IDX_37[aatype_for_index]

    # Avoid to_tensor_7() which causes NaN gradients due to quaternion conversion singularity.
    # Instead, gather directly on rotation matrices and translations.
    rots = r.get_rots().get_rot_mats()  # [..., N, 8, 3, 3]
    trans = r.get_trans()               # [..., N, 8, 3]

    # Gather selected frames for each atom: [..., N, 37] -> [..., N, 37, 3, 3]
    gather_idx_rot = group_idx.unsqueeze(-1).unsqueeze(-1).expand(*group_idx.shape, 3, 3).to(rots.device)
    selected_rots = torch.gather(rots, -3, gather_idx_rot)

    # Gather selected translations: [..., N, 37] -> [..., N, 37, 3]
    gather_idx_trans = group_idx.unsqueeze(-1).expand(*group_idx.shape, 3).to(trans.device)
    selected_trans = torch.gather(trans, -2, gather_idx_trans)

    selected_frames = Rigid(Rotation(rot_mats=selected_rots), selected_trans)

    frame_null_pos = IDEALIZED_POS_37[aatype_for_index].to(aatype_device)
    pred_positions = selected_frames.apply(frame_null_pos)
    frame_atom_mask = ATOM_MASK_37[aatype_for_index].to(aatype_device).unsqueeze(-1)
    pred_positions = pred_positions * frame_atom_mask
    return pred_positions

def calculate_neighbor_angles(R_ac, R_ab):
    """Calculate angles between atoms c <- a -> b.

    Parameters
    ----------
        R_ac: Tensor, shape = (N,3)
            Vector from atom a to c.
        R_ab: Tensor, shape = (N,3)
            Vector from atom a to b.

    Returns
    -------
        angle_cab: Tensor, shape = (N,)
            Angle between atoms c <- a -> b.
    """
    # cos(alpha) = (u * v) / (|u|*|v|)
    x = torch.sum(R_ac * R_ab, dim=1)  # shape = (N,)
    # sin(alpha) = |u x v| / (|u|*|v|)
    y = torch.cross(R_ac, R_ab).norm(dim=-1)  # shape = (N,)
    # avoid that for y == (0,0,0) the gradient wrt. y becomes NaN
    y = torch.max(y, torch.tensor(1e-9))  
    angle = torch.atan2(y, x)
    return angle


def vector_projection(R_ab, P_n):
    """
    Project the vector R_ab onto a plane with normal vector P_n.

    Parameters
    ----------
        R_ab: Tensor, shape = (N,3)
            Vector from atom a to b.
        P_n: Tensor, shape = (N,3)
            Normal vector of a plane onto which to project R_ab.

    Returns
    -------
        R_ab_proj: Tensor, shape = (N,3)
            Projected vector (orthogonal to P_n).
    """
    a_x_b = torch.sum(R_ab * P_n, dim=-1)
    b_x_b = torch.sum(P_n * P_n, dim=-1)
    return R_ab - (a_x_b / b_x_b)[:, None] * P_n


# ============================================================================
# HYBRID PROTEIN-RNA CONVERTER - ADDED FOR RNA IPA SUPPORT
# ============================================================================

from typing import Dict, Tuple, Optional
from src.data.residue_constants import (
    is_rna_residue, is_protein_residue,
    rna_restypes, RNA_ATOM_NUM_MAX, hybrid_restype_num
)


class HybridConverter:
    """Unified converter for protein-RNA hybrid structures.
    
    Handles conversion between rigids and all-atom coordinates for both
    protein and RNA residues in a unified framework.
    """
    
    def __init__(self, device=None):
        """Initialize hybrid converter.
        
        Args:
            device: Torch device to use
        """
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._init_rna_converter()
    
    def _init_rna_converter(self):
        """Initialize RNA converter if RhoFold is available."""
        try:
            from rhofold.utils.converter import RNAConverter
            self.rna_converter = RNAConverter()
            self.has_rhofold = True
        except ImportError:
            self.rna_converter = None
            self.has_rhofold = False
            print("Warning: RhoFold not available, RNA conversion will be limited")
    
    def rigids_to_atom37(
        self,
        rigids: Rigid,
        aatype: torch.Tensor,
        is_rna_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert rigids to unified atom coordinates for hybrid structures.
        
        Args:
            rigids: [..., N] Rigid objects (backbone frames)
            aatype: [..., N] residue type indices (0-31, where 21-25 are RNA: A=21, G=22, C=23, U=24, N=25)
            is_rna_mask: [..., N] optional bool mask for RNA residues
            
        Returns:
            atom_pos: [..., N, 37, 3] unified atom positions
            atom_mask: [..., N, 37] unified atom mask
        """
        # Detect RNA residues if mask not provided
        # Use vectorized operations to avoid GPU->CPU->GPU transfer
        if is_rna_mask is None:
            is_rna_mask = (aatype >= 21) & (aatype <= 25)
        
        is_protein_mask = ~is_rna_mask
        
        # Initialize output tensors with unified size (37 atoms)
        batch_shape = aatype.shape
        
        atom_pos = torch.zeros(*batch_shape, 37, 3, device=aatype.device)
        atom_mask = torch.zeros(*batch_shape, 37, device=aatype.device)
        
        # Process protein residues
        if is_protein_mask.any():
            prot_pos, prot_mask = self._convert_protein(
                rigids, aatype, is_protein_mask
            )
            # Ensure protein output is [..., N, 37, 3]
            if prot_pos.shape[-2] < 37:
                prot_pos = torch.nn.functional.pad(prot_pos, (0, 0, 0, 37 - prot_pos.shape[-2]))
            if prot_mask.shape[-1] < 37:
                prot_mask = torch.nn.functional.pad(prot_mask, (0, 37 - prot_mask.shape[-1]))
            
            atom_pos = torch.where(
                is_protein_mask[..., None, None],
                prot_pos[..., :37, :],
                atom_pos
            )
            atom_mask = torch.where(
                is_protein_mask[..., None],
                prot_mask[..., :37],
                atom_mask
            )
        
        # Process RNA residues
        if is_rna_mask.any():
            rna_pos, rna_mask = self._convert_rna(
                rigids, aatype, is_rna_mask
            )
            # Pad RNA output [..., N, 23, 3] to [..., N, 37, 3]
            if rna_pos.shape[-2] < 37:
                rna_pos = torch.nn.functional.pad(rna_pos, (0, 0, 0, 37 - rna_pos.shape[-2]))
            if rna_mask.shape[-1] < 37:
                rna_mask = torch.nn.functional.pad(rna_mask, (0, 37 - rna_mask.shape[-1]))
            
            atom_pos = torch.where(
                is_rna_mask[..., None, None],
                rna_pos[..., :37, :],
                atom_pos
            )
            atom_mask = torch.where(
                is_rna_mask[..., None],
                rna_mask[..., :37],
                atom_mask
            )
        
        return atom_pos, atom_mask
    
    def _convert_protein(
        self,
        rigids: Rigid,
        aatype: torch.Tensor,
        protein_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert protein rigids to atom37 coordinates.
        
        Uses existing OpenFold protein conversion logic.
        """
        # Use protein aatype (0-19), clamp to valid range and ensure same device as constants
        prot_aatype = torch.where(protein_mask, aatype, 0).long()
        # Clamp to valid protein indices (0-19) to prevent out-of-bounds indexing
        prot_aatype = torch.clamp(prot_aatype, 0, 19)
        
        # Default torsion angles (phi=psi=omega=0, chi=0)
        torsion_angles = torch.zeros(
            *aatype.shape, 7, 2, device=aatype.device
        )
        torsion_angles[..., 0] = 1.0  # cos=1, sin=0 for all angles
        
        # Convert to frames
        all_frames = torsion_angles_to_frames(
            rigids, torsion_angles, prot_aatype
        )
        
        # Convert frames to atom positions
        atom37_pos = frames_to_atom37_pos(all_frames, prot_aatype)
        
        # Get mask from residue_constants - ensure device match for indexing
        prot_aatype_for_mask = prot_aatype.to(device=ATOM_MASK_37.device).long()
        atom37_mask = ATOM_MASK_37[prot_aatype_for_mask].to(prot_aatype.device)
        
        # Zero out non-protein positions
        atom37_pos = atom37_pos * protein_mask[..., None, None]
        atom37_mask = atom37_mask * protein_mask[..., None]
        
        return atom37_pos, atom37_mask
    
    def _convert_rna(
        self,
        rigids: Rigid,
        aatype: torch.Tensor,
        rna_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert RNA rigids to atom23 coordinates.
        
        Uses RhoFold RNA conversion logic.
        """
        if not self.has_rhofold:
            # Fallback: return dummy coordinates
            batch_shape = aatype.shape
            return (
                torch.zeros(*batch_shape, 23, 3, device=aatype.device),
                torch.zeros(*batch_shape, 23, device=aatype.device)
            )
        
        # Map aatype to RNA indices (21-24 -> 0-3 for RhoFold)
        # Protenix: A=21, G=22, C=23, U=24, N=25 (unused)
        # RhoFold internal: A=0, G=1, U=2, C=3
        protenix_to_rhofold = {21: 0, 22: 1, 23: 3, 24: 2}
        rna_aatype = torch.zeros_like(aatype)
        for p_idx, r_idx in protenix_to_rhofold.items():
            rna_aatype = torch.where(aatype == p_idx, r_idx, rna_aatype)
        
        # Convert rigids to tensor_7 format [N, 7]
        frames_t7 = rigids.to_tensor_7()  # [..., N, 7]
        
        # Build RNA sequence string (RhoFold order: A, G, U, C)
        # Handle batch dimensions properly
        original_shape = aatype.shape
        rhofold_to_seq = {0: 'A', 1: 'G', 2: 'U', 3: 'C'}
        
        # Flatten batch dimensions for sequence construction
        rna_aatype_flat = rna_aatype.reshape(-1)
        seq = ''.join([rhofold_to_seq[int(i)] for i in rna_aatype_flat.cpu().numpy()])
        
        # Default RNA torsion angles (6 angles per residue)
        # RhoFold expects: [omega, phi, angl_0, angl_1, angl_2, angl_3] each with [cos, sin]
        total_residues = rna_aatype_flat.shape[0]
        angles = torch.zeros(total_residues, 6, 2, device=aatype.device)
        angles[..., 0] = 1.0  # cos=1, sin=0 for all angles (identity)
        
        # Build coordinates using RhoFold converter
        # Note: This processes all residues but we only use RNA positions
        try:
            cord, cmsk = self.rna_converter.build_cords(
                seq, frames_t7, angles, rtn_cmsk=True
            )
            
            # Reshape back to original batch dimensions
            cord = cord.reshape(*original_shape, 23, 3)
            cmsk = cmsk.reshape(*original_shape, 23)
            
        except Exception as e:
            raise RuntimeError(f"RNA coordinate conversion failed: {e}") from e
        
        # Zero out non-RNA positions
        cord = cord * rna_mask[..., None, None]
        cmsk = cmsk * rna_mask[..., None]
        
        return cord, cmsk


def hybrid_rigids_to_allatom(
    rigids: Rigid,
    aatype: torch.Tensor,
    is_rna_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified function to convert rigids to all-atom coordinates.
    
    This is a convenience wrapper around HybridConverter.
    
    Args:
        rigids: [..., N] Rigid objects
        aatype: [..., N] residue type indices (0-23)
        is_rna_mask: [..., N] optional bool mask (auto-detected if None)
        
    Returns:
        atom_pos: [..., N, max_atoms, 3] (max_atoms = 37)
        atom_mask: [..., N, max_atoms]
    """
    converter = HybridConverter(device=aatype.device)
    return converter.rigids_to_atom37(rigids, aatype, is_rna_mask)


def get_backbone_atoms_for_hybrid(
    aatype: torch.Tensor,
    atom_pos: torch.Tensor,
    is_rna_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Get backbone atom positions for hybrid protein-RNA structures.
    
    For protein: returns CA positions (index 1 in atom37)
    For RNA: returns C4' positions (index 0 in atom23)
    
    Args:
        aatype: [N] residue type indices
        atom_pos: [N, num_atoms, 3] atom positions
        is_rna_mask: [N] optional RNA mask
        
    Returns:
        bb_pos: [N, 3] backbone positions
    """
    if is_rna_mask is None:
        # Use vectorized operations to avoid GPU->CPU->GPU transfer
        is_rna_mask = (aatype >= 21) & (aatype <= 25)
    
    # Protein: CA is at index 1 in atom37
    # RNA: C4' is at index 0 in atom23
    prot_bb = atom_pos[..., 1, :]  # CA
    rna_bb = atom_pos[..., 0, :]   # C4'
    
    return torch.where(is_rna_mask[..., None], rna_bb, prot_bb)
