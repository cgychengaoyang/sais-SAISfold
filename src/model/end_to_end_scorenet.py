"""End-to-end model: Protenix-style PairFormer + SE3 diffusion."""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

from src.model.score_based_ipa import DyneTrionScoreNet
from src.model.protenix.model.modules.embedders import (
    InputFeatureEmbedder,
    RelativePositionEncoding,
)
from src.model.protenix.model.modules.pairformer import PairformerStack
from src.model.protenix.model.modules.primitives import LinearNoBias
from src.model.protenix.openfold_local.model.primitives import LayerNorm


def build_minimal_protenix_features(
    aatype: torch.Tensor,
    seq_idx: Optional[torch.Tensor] = None,
    num_atoms_per_token: int = 1,
) -> Dict[str, torch.Tensor]:
    """Build minimal Protenix-compatible input features from aatype.
    
    Args:
        aatype: [B, N] or [N] amino acid type indices (0-19 for protein)
        seq_idx: [B, N] or [N] sequence indices. If None, uses 0..N-1.
        num_atoms_per_token: Number of atoms per residue token (default 1 for minimal)
    
    Returns:
        Dictionary of Protenix input features.
    """
    if aatype.dim() == 1:
        aatype = aatype.unsqueeze(0)
    batch_shape = aatype.shape[:-1]
    N = aatype.shape[-1]
    device = aatype.device
    
    # Position features
    if seq_idx is None:
        seq_idx = torch.arange(N, device=device)
    if seq_idx.dim() == 1:
        seq_idx = seq_idx.unsqueeze(0).expand(*batch_shape, -1)
    
    # Simple features: single chain, single entity
    asym_id = torch.zeros(*batch_shape, N, dtype=torch.long, device=device)
    residue_index = seq_idx.long()
    entity_id = torch.zeros(*batch_shape, N, dtype=torch.long, device=device)
    sym_id = torch.zeros(*batch_shape, N, dtype=torch.long, device=device)
    token_index = torch.arange(N, device=device).unsqueeze(0).expand(*batch_shape, -1)
    
    # Biomolecule flags (all protein)
    is_protein = torch.ones(*batch_shape, N, 1, device=device)
    is_rna = torch.zeros(*batch_shape, N, 1, device=device)
    is_dna = torch.zeros(*batch_shape, N, 1, device=device)
    is_ligand = torch.zeros(*batch_shape, N, 1, device=device)
    
    # Restype one-hot (32 classes: 20 AA + UNK + GAP + 4 RNA + 4 DNA + 4 special)
    restype = torch.nn.functional.one_hot(aatype, num_classes=32).float()
    
    # Dummy MSA features
    profile = torch.zeros(*batch_shape, N, 32, device=device)
    deletion_mean = torch.zeros(*batch_shape, N, 1, device=device)
    
    # Atom features: use OpenFold idealized positions for up to 5 atoms per residue
    from openfold.np import residue_constants as rc
    
    # Select first 5 atoms from atom37 ordering: N, CA, C, CB, O
    atom37_order = ['N', 'CA', 'C', 'CB', 'O']
    num_atoms_per_token = min(num_atoms_per_token, len(atom37_order))
    
    total_atoms = np.prod(batch_shape) * N * num_atoms_per_token
    atom_to_token_idx = torch.arange(total_atoms, device=device) // num_atoms_per_token
    atom_to_token_idx = atom_to_token_idx.reshape(*batch_shape, N * num_atoms_per_token)
    
    # Get reference positions from OpenFold constants [21 restypes, 37 atoms, 3]
    ref_pos = torch.zeros(*batch_shape, N * num_atoms_per_token, 3, device=device)
    for atom_idx, atom_name in enumerate(atom37_order[:num_atoms_per_token]):
        atom_type_idx = rc.atom_order[atom_name]
        # Gather positions for each residue
        positions = torch.from_numpy(rc.restype_atom37_rigid_group_positions[aatype.cpu().numpy(), atom_type_idx]).float()  # [B, N, 3]
        ref_pos[..., atom_idx::num_atoms_per_token, :] = positions.to(device)
    
    ref_charge = torch.zeros(*batch_shape, N * num_atoms_per_token, 1, device=device)
    ref_mask = torch.ones(*batch_shape, N * num_atoms_per_token, device=device)
    
    # Element one-hot: N=7, C=6, O=8
    ref_element = torch.zeros(*batch_shape, N * num_atoms_per_token, 128, device=device)
    element_map = {'N': 6, 'C': 5, 'O': 7}  # zero-indexed: H=0, C=5, N=6, O=7
    for atom_idx, atom_name in enumerate(atom37_order[:num_atoms_per_token]):
        elem = element_map.get(atom_name[0], 5)
        ref_element[..., atom_idx::num_atoms_per_token, elem] = 1.0
    
    ref_atom_name_chars = torch.zeros(*batch_shape, N * num_atoms_per_token, 4, 64, device=device)
    
    # ref_space_uid for dense trunk computation
    ref_space_uid = atom_to_token_idx.clone()
    
    # Token bonds (no special bonds)
    token_bonds = torch.zeros(*batch_shape, N, N, device=device)
    
    features = {
        "aatype": aatype,
        "asym_id": asym_id,
        "residue_index": residue_index,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "token_index": token_index,
        "is_protein": is_protein,
        "is_rna": is_rna,
        "is_dna": is_dna,
        "is_ligand": is_ligand,
        "restype": restype,
        "profile": profile,
        "deletion_mean": deletion_mean,
        "atom_to_token_idx": atom_to_token_idx,
        "ref_pos": ref_pos,
        "ref_charge": ref_charge,
        "ref_mask": ref_mask,
        "ref_element": ref_element,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": ref_space_uid,
        "token_bonds": token_bonds,
    }
    
    # Generate dense trunk features (d_lm, v_lm, pad_info)
    from src.model.protenix.model.modules.transformer import rearrange_qk_to_dense_trunk
    with torch.no_grad():
        q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(
            q=[features["ref_pos"], features["ref_space_uid"]],
            k=[features["ref_pos"], features["ref_space_uid"]],
            dim_q=[-2, -1],
            dim_k=[-2, -1],
            n_queries=32,
            n_keys=128,
            compute_mask=True,
        )
        d_lm = q_trunked_list[0][..., None, :] - k_trunked_list[0][..., None, :, :]
        v_lm = (q_trunked_list[1][..., None].int() == k_trunked_list[1][..., None, :].int()).unsqueeze(dim=-1)
        features["d_lm"] = d_lm
        features["v_lm"] = v_lm
        features["pad_info"] = pad_info
    
    # Generate relative position encoding
    relpe = RelativePositionEncoding(r_max=32, s_max=2, c_z=128)
    if relpe.training:
        relpe.eval()
    with torch.no_grad():
        features = relpe.generate_relp(features)
    relpe.to(device)
    
    return features


class ProtenixPairformerEmbedder(nn.Module):
    """Computes node and edge embeddings using Protenix-style PairFormer."""
    
    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        pairformer_blocks: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_inputs = c_s_inputs
        
        self.input_embedder = InputFeatureEmbedder(
            c_atom=128,
            c_atompair=16,
            c_token=384,
            esm_configs={},
        )
        self.relative_position_encoding = RelativePositionEncoding(
            r_max=32, s_max=2, c_z=c_z
        )
        self.pairformer_stack = PairformerStack(
            n_blocks=pairformer_blocks,
            n_heads=16,
            c_z=c_z,
            c_s=c_s,
            dropout=dropout,
            blocks_per_ckpt=None,
        )
        
        self.linear_no_bias_sinit = LinearNoBias(c_s_inputs, c_s)
        self.linear_no_bias_zinit1 = LinearNoBias(c_s, c_z)
        self.linear_no_bias_zinit2 = LinearNoBias(c_s, c_z)
        self.linear_no_bias_token_bond = LinearNoBias(1, c_z)
        self.linear_no_bias_z_cycle = LinearNoBias(c_z, c_z)
        self.linear_no_bias_s = LinearNoBias(c_s, c_s)
        self.layernorm_z_cycle = LayerNorm(c_z)
        self.layernorm_s = LayerNorm(c_s)
        
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)
    
    def forward(self, input_feature_dict: Dict[str, torch.Tensor]) -> tuple:
        """Returns node_repr [..., N, c_s] and edge_repr [..., N, N, c_z]."""
        s_inputs = self.input_embedder(input_feature_dict)
        s_init = self.linear_no_bias_sinit(s_inputs)
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )
        z_init = z_init + self.relative_position_encoding(input_feature_dict["relp"])
        z_init = z_init + self.linear_no_bias_token_bond(
            input_feature_dict["token_bonds"].unsqueeze(dim=-1)
        )
        
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)
        
        for cycle_no in range(1):  # Single cycle for efficiency
            z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
            s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
            s, z = self.pairformer_stack(
                s,
                z,
                pair_mask=None,
                triangle_multiplicative="torch",
                triangle_attention="torch",
                inplace_safe=False,
                chunk_size=None,
            )
        
        return s, z


class EndToEndDyneTrionScoreNet(nn.Module):
    """End-to-end model: Protenix PairFormer embedder + DyneTrion SE3 diffusion."""
    
    def __init__(
        self,
        c_s_input: int = 384,
        c_z_input: int = 128,
        c_s: int = 256,
        c_z: int = 128,
        num_blocks: int = 4,
        pairformer_blocks: int = 4,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
        ipa_config: Optional[Dict] = None,
        coordinate_scaling: float = 0.1,
    ):
        super().__init__()
        self.embedder = ProtenixPairformerEmbedder(
            c_s=c_s_input,
            c_z=c_z_input,
            pairformer_blocks=pairformer_blocks,
            dropout=dropout,
        )
        self.score_net = DyneTrionScoreNet(
            c_s_input=c_s_input,
            c_z_input=c_z_input,
            c_s=c_s,
            c_z=c_z,
            num_blocks=num_blocks,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            ipa_config=ipa_config,
            coordinate_scaling=coordinate_scaling,
        )
    
    def set_diffuser(self, diffuser):
        self.score_net.set_diffuser(diffuser)
    
    def forward(
        self,
        aatype: torch.Tensor,
        rigids: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        input_feature_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass from raw inputs to structure predictions.
        
        Args:
            aatype: [B, N] amino acid type indices
            rigids: [B, N, 7] initial rigids
            t: [B] diffusion timestep
            mask: [B, N] optional mask
            seq_idx: [B, N] optional sequence indices
            input_feature_dict: Optional pre-built Protenix input features.
                                If None, builds minimal features from aatype.
        
        Returns:
            Same outputs as DyneTrionScoreNet.forward()
        """
        # Build or use provided Protenix features
        if input_feature_dict is None:
            input_features = build_minimal_protenix_features(aatype, seq_idx)
        else:
            input_features = input_feature_dict
        
        # Compute embeddings end-to-end
        node_repr, edge_repr = self.embedder(input_features)
        
        # Add batch dim if needed (embedder returns [N, C] for single sample)
        if node_repr.dim() == 2:
            node_repr = node_repr.unsqueeze(0)
            edge_repr = edge_repr.unsqueeze(0)
        
        # Run SE3 diffusion model
        return self.score_net(node_repr, edge_repr, rigids, t, mask)
    
    def sample_structure(
        self,
        aatype: torch.Tensor,
        num_steps: int = 100,
        mask: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        trans_scale: float = 10.0,
    ) -> torch.Tensor:
        """Sample structure from raw sequence inputs."""
        B, N = aatype.shape[:2]
        device = aatype.device
        
        rigids = DyneTrionScoreNet.init_random_rigids(B, N, trans_scale, device)
        
        for i in range(num_steps):
            t = torch.ones(B, device=device) * (1 - i / num_steps)
            with torch.no_grad():
                out = self.forward(aatype, rigids, t, mask, seq_idx)
            
            rot_score = out['pred_rot_score']
            trans_score = out['pred_trans_score']
            step_size = 0.01 * (1 - i / num_steps)
            
            rigids[..., 4:] = rigids[..., 4:] + step_size * trans_score
            rot_update = step_size * rot_score
            from src.model.score_based_ipa import rotvec_to_quat, quat_multiply
            rot_quat = rotvec_to_quat(rot_update)
            curr_quat = rigids[..., :4]
            new_quat = quat_multiply(rot_quat, curr_quat)
            new_quat = new_quat / new_quat.norm(dim=-1, keepdim=True)
            rigids[..., :4] = new_quat
        
        return rigids
