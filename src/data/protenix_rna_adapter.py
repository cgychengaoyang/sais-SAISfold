"""Protenix-compatible RNA feature adapter.

This module provides utilities for RNA features compatible with
Protenix/AlphaFold3-style input format.

Note: Since DyneTrion now uses the same restype encoding as Protenix
(A=22, C=23, G=24, U=25), no conversion is needed for restype.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


# RNA atom element types (atomic numbers)
RNA_ATOM_ELEMENTS = {
    "C4'": 6, "C1'": 6, 'N9': 7, 'N1': 7,
    "C2'": 6, "C3'": 6, "C5'": 6,
    "O2'": 8, "O3'": 8, "O4'": 8,
    'N2': 7, 'N3': 7, 'N4': 7, 'N6': 7, 'N7': 7,
    'C2': 6, 'C4': 6, 'C5': 6, 'C6': 6, 'C8': 6,
    'O2': 8, 'O4': 8, 'O6': 8,
    "O5'": 8, 'P': 15, 'OP1': 8, 'OP2': 8,
}

# RNA atom charges (approximate)
RNA_ATOM_CHARGES = {
    "C4'": 0.0, "C1'": 0.0, 'N9': 0.0, 'N1': 0.0,
    "C2'": 0.0, "C3'": 0.0, "C5'": 0.0,
    "O2'": -0.5, "O3'": -0.5, "O4'": -0.5,
    'N2': 0.0, 'N3': 0.0, 'N4': 0.0, 'N6': 0.0, 'N7': 0.0,
    'C2': 0.0, 'C4': 0.0, 'C5': 0.0, 'C6': 0.0, 'C8': 0.0,
    'O2': -0.5, 'O4': -0.5, 'O6': -0.5,
    "O5'": -0.5, 'P': 1.5, 'OP1': -0.75, 'OP2': -0.75,
}


class ProtenixRNAAdapter:
    """Adapter for Protenix-compatible RNA features.
    
    Note: DyneTrion now uses the same restype encoding as Protenix,
    so no conversion is needed for restype indices.
    """
    
    def __init__(self, device=None):
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize RNA reference positions from RhoFold constants
        self._init_rna_ref_positions()
    
    def _init_rna_ref_positions(self):
        """Initialize RNA reference positions from RhoFold constants.
        
        Note: RhoFold's ATOM_INFOS_PER_RESD and ATOM_NAMES_PER_RESD may have
        different orderings. We use ATOM_NAMES_PER_RESD as the canonical ordering
        and map coordinates from ATOM_INFOS_PER_RESD by atom name.
        """
        try:
            from rhofold.utils.constants import RNA_CONSTANTS
            
            # Build reference position tensor [4 restypes, 23 atoms, 3]
            self.rna_ref_positions = torch.zeros(4, 23, 3, device=self.device)
            self.rna_ref_mask = torch.zeros(4, 23, dtype=torch.bool, device=self.device)
            
            # Map Protenix restype indices to RhoFold order
            # Protenix: A=22, C=23, G=24, U=25
            # RhoFold internal: A=0, C=3, G=1, U=2
            restype_order = ['A', 'G', 'U', 'C']  # RhoFold internal order
            self.protenix_to_rhofold = {22: 0, 23: 3, 24: 1, 25: 2}
            self.rhofold_to_protenix = {0: 22, 3: 23, 1: 24, 2: 25}
            
            for restype_name in restype_order:
                rhofold_idx = restype_order.index(restype_name)
                
                # Get canonical atom ordering from ATOM_NAMES_PER_RESD
                atom_names = RNA_CONSTANTS.ATOM_NAMES_PER_RESD[restype_name]
                
                # Build lookup from atom_name to coordinates from ATOM_INFOS_PER_RESD
                atom_info_dict = {}
                for atom_info in RNA_CONSTANTS.ATOM_INFOS_PER_RESD[restype_name]:
                    atom_name, _, coords = atom_info
                    atom_info_dict[atom_name] = coords
                
                # Populate positions in canonical order
                for atom_idx, atom_name in enumerate(atom_names):
                    if atom_idx < 23 and atom_name in atom_info_dict:
                        coords = atom_info_dict[atom_name]
                        self.rna_ref_positions[rhofold_idx, atom_idx] = torch.tensor(
                            coords, device=self.device, dtype=torch.float32
                        )
                        self.rna_ref_mask[rhofold_idx, atom_idx] = True
            
            self.rna_atom_names = RNA_CONSTANTS.ATOM_NAMES_PER_RESD
            self.has_rhofold = True
            
        except ImportError:
            # Fallback: use default positions
            self.rna_ref_positions = torch.zeros(4, 23, 3, device=self.device)
            self.rna_ref_mask = torch.ones(4, 23, dtype=torch.bool, device=self.device)
            self.protenix_to_rhofold = {22: 0, 23: 3, 24: 1, 25: 2}
            self.rhofold_to_protenix = {0: 22, 3: 23, 1: 24, 2: 25}
            self.rna_atom_names = {
                'A': ["C4'", "C1'", 'N9', "C2'", "C3'", "C5'", "O2'", "O3'", "O4'", 
                      'N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N6', 'N7', 'C8', "O5'", 'P', 'OP1', 'OP2'],
                'G': ["C4'", "C1'", 'N9', "C2'", "C3'", "C5'", "O2'", "O3'", "O4'",
                      'N1', 'N2', 'N3', 'C2', 'C4', 'C5', 'C6', 'N7', 'C8', 'O6', "O5'", 'P', 'OP1', 'OP2'],
                'U': ["C4'", "C1'", 'N1', "C2'", "C3'", "C5'", "O2'", "O3'", "O4'",
                      'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6', "O5'", 'P', 'OP1', 'OP2'],
                'C': ["C4'", "C1'", 'N1', "C2'", "C3'", "C5'", "O2'", "O3'", "O4'",
                      'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6', "O5'", 'P', 'OP1', 'OP2'],
            }
            self.has_rhofold = False
    
    def create_biomolecule_flags(self, aatype: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create Protenix biomolecule type flags.
        
        Args:
            aatype: Residue type indices (0-25, RNA at 22-25)
            
        Returns:
            Dictionary with 'is_protein', 'is_rna', 'is_dna', 'is_ligand'
            Each has shape [..., 1] with float values 0.0 or 1.0
        """
        # Protein: 0-19
        is_protein = ((aatype >= 0) & (aatype <= 19)).float()
        
        # RNA: 22-25
        is_rna = ((aatype >= 22) & (aatype <= 25)).float()
        
        # DNA: 26-29 (not used currently)
        is_dna = ((aatype >= 26) & (aatype <= 29)).float()
        
        # Ligand: 30+
        is_ligand = (aatype >= 30).float()
        
        # Add dimension for shape [..., 1]
        return {
            'is_protein': is_protein[..., None],
            'is_rna': is_rna[..., None],
            'is_dna': is_dna[..., None],
            'is_ligand': is_ligand[..., None],
        }
    
    def create_restype_onehot(self, aatype: torch.Tensor) -> torch.Tensor:
        """Create Protenix-compatible 32-dim restype one-hot encoding.
        
        Args:
            aatype: Residue type indices (0-25, RNA at 22-25)
            
        Returns:
            One-hot encoding [..., 32]
        """
        # Protenix uses 32-dim one-hot
        # 0-19: protein, 20: UNK, 21: GAP, 22-25: RNA, 26-29: DNA, 30-31: special
        return torch.nn.functional.one_hot(aatype, num_classes=32).float()
    
    def create_rna_ref_features(
        self,
        aatype: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Create Protenix-compatible reference features for RNA.
        
        Args:
            aatype: Residue type indices (0-25, RNA at 22-25)
            
        Returns:
            Dictionary with reference features
        """
        batch_shape = aatype.shape
        device = aatype.device
        
        # Detect RNA residues (indices 22-25)
        is_rna = ((aatype >= 22) & (aatype <= 25))
        
        # Map Protenix indices to RhoFold indices (0-3)
        rhofold_indices = torch.zeros_like(aatype)
        for p_idx, r_idx in self.protenix_to_rhofold.items():
            rhofold_indices = torch.where(aatype == p_idx, r_idx, rhofold_indices)
        
        # Get reference positions (on correct device)
        ref_pos = self.rna_ref_positions.to(device)[rhofold_indices]  # [..., 23, 3]
        ref_mask = self.rna_ref_mask.to(device)[rhofold_indices]  # [..., 23]
        
        # Only valid for RNA residues
        ref_mask = ref_mask & is_rna[..., None]
        
        # Create placeholder features (simplified)
        ref_charge = torch.zeros(*batch_shape, 23, device=device)
        ref_element = torch.zeros(*batch_shape, 23, 128, device=device)
        ref_atom_name_chars = torch.zeros(*batch_shape, 23, 4, 64, device=device)
        
        # Create atom-to-token mapping
        total_atoms = np.prod(batch_shape) * 23
        atom_to_token_idx = torch.arange(total_atoms, device=device) // 23
        atom_to_token_idx = atom_to_token_idx.reshape(*batch_shape, 23)
        
        return {
            'ref_pos': ref_pos,
            'ref_mask': ref_mask.float(),
            'ref_charge': ref_charge,
            'ref_element': ref_element,
            'ref_atom_name_chars': ref_atom_name_chars,
            'atom_to_token_idx': atom_to_token_idx,
        }
    
    def create_dummy_msa_features(self, aatype: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create dummy MSA features for RNA (required by Protenix).
        
        RNA typically doesn't have MSA, so these are zeros.
        
        Args:
            aatype: Residue type indices
            
        Returns:
            Dictionary with 'profile' and 'deletion_mean'
        """
        batch_shape = aatype.shape
        device = aatype.device
        
        # Profile: 32-dim (same as restype one-hot dimension)
        profile = torch.zeros(*batch_shape, 32, device=device)
        
        # For RNA residues, set profile to match restype
        is_rna = ((aatype >= 22) & (aatype <= 25))
        if is_rna.any():
            restype_onehot = self.create_restype_onehot(aatype)
            profile = torch.where(is_rna[..., None], restype_onehot, profile)
        
        # Deletion mean: 1-dim, zeros for RNA
        deletion_mean = torch.zeros(*batch_shape, 1, device=device)
        
        return {
            'profile': profile,
            'deletion_mean': deletion_mean,
        }
    
    def create_protenix_features(
        self,
        aatype: torch.Tensor,
        extra_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Create complete Protenix input feature dict.
        
        Args:
            aatype: Residue type indices (0-25, RNA at 22-25)
            extra_features: Optional additional features
            
        Returns:
            Complete Protenix-compatible feature dictionary
        """
        features = {}
        
        # Biomolecule type flags
        features.update(self.create_biomolecule_flags(aatype))
        
        # Restype one-hot
        features['restype'] = self.create_restype_onehot(aatype)
        
        # MSA features (dummy for RNA)
        features.update(self.create_dummy_msa_features(aatype))
        
        # Reference features for RNA
        rna_ref_features = self.create_rna_ref_features(aatype)
        features.update(rna_ref_features)
        
        # Add any extra features
        if extra_features is not None:
            features.update(extra_features)
        
        return features


def create_protenix_biomolecule_flags(aatype):
    """Standalone function to create biomolecule flags.
    
    Args:
        aatype: Residue type indices (int, list, or tensor)
        
    Returns:
        Dictionary with is_protein, is_rna, is_dna, is_ligand
    """
    if isinstance(aatype, torch.Tensor):
        adapter = ProtenixRNAAdapter(device=aatype.device)
        return adapter.create_biomolecule_flags(aatype)
    else:
        import numpy as np
        aatype_arr = np.array(aatype)
        
        is_protein = ((aatype_arr >= 0) & (aatype_arr <= 19)).astype(float)
        is_rna = ((aatype_arr >= 22) & (aatype_arr <= 25)).astype(float)
        is_dna = ((aatype_arr >= 26) & (aatype_arr <= 29)).astype(float)
        is_ligand = (aatype_arr >= 30).astype(float)
        
        return {
            'is_protein': is_protein,
            'is_rna': is_rna,
            'is_dna': is_dna,
            'is_ligand': is_ligand,
        }
