"""Simplified Seq -> Structure IPA Model.

This is a simplified version of DyneTrion that removes all temporal components
and focuses purely on sequence-to-structure prediction like standard protein
folding models (AlphaFold2/3, Protenix).

Workflow:
    Input: node_repr, edge_repr (from Protenix/GeoForm)
    -> IPA Blocks (no temporal) -> Rigid Updates -> Angles -> Coordinates
    
Key differences from DyneTrion:
    - No motion_rigids, ref_rigids, frame_time dimensions
    - No temporal transformers
    - No ReferenceNet spatial alignment
    - Single structure output instead of trajectory
    - Simplified diffusion (optional timestep embedding)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.model.structure_module import (
    InvariantPointAttention,
    StructureModuleTransition,
    BackboneUpdate,
    AngleResnet,
    EdgeTransition,
)


class SeqStructureIPABlock(nn.Module):
    """Single IPA block for sequence-to-structure prediction.
    
    Simplified from DyneTrion - no temporal or reference components.
    """
    
    def __init__(
        self,
        c_s: int = 256,
        c_z: int = 128,
        c_ipa: int = 16,
        no_heads: int = 8,
        no_qk_points: int = 4,
        no_v_points: int = 8,
        dropout: float = 0.0,
        use_edge_update: bool = True,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.use_edge_update = use_edge_update
        
        # Invariant Point Attention - core of structure prediction
        self.ipa = InvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_ipa,
            no_heads=no_heads,
            no_qk_points=no_qk_points,
            no_v_points=no_v_points,
        )
        self.ipa_dropout = nn.Dropout(dropout)
        self.ipa_layer_norm = nn.LayerNorm(c_s)
        
        # Node transition (MLP on node features)
        self.node_transition = StructureModuleTransition(c_s)
        
        # Backbone update - predicts rigid transformations
        self.bb_update = BackboneUpdate(c_s)
        
        # Edge transition (optional)
        if use_edge_update:
            self.edge_transition = EdgeTransition(
                c_s=c_s,
                c_z=c_z,
            )
    
    def forward(
        self,
        s: torch.Tensor,        # [B, N, c_s] node features
        z: torch.Tensor,        # [B, N, N, c_z] edge features
        r: Rigid,               # [B, N] current rigids
        mask: torch.Tensor,     # [B, N] residue mask
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Rigid]:
        """Single forward pass through IPA block.
        
        Args:
            s: Node features [B, N, c_s]
            z: Edge features [B, N, N, c_z]
            r: Current rigids [B, N]
            mask: Residue mask [B, N]
            
        Returns:
            s: Updated node features
            z: Updated edge features (or None if no edge update)
            r: Updated rigids
        """
        # IPA attention with geometric invariance
        ipa_out = self.ipa(s, z, r, mask)
        ipa_out = self.ipa_dropout(ipa_out)
        s = self.ipa_layer_norm(s + ipa_out)
        
        # Node transition
        s = self.node_transition(s)
        
        # Backbone update - predict SE(3) transformation
        # Output: [B, N, 6] -> 3 for rotation (quat update), 3 for translation
        rigid_update = self.bb_update(s)
        
        # Apply update to rigids
        # compose_q_update_vec handles the composition
        r = r.compose_q_update_vec(rigid_update, mask[..., None])
        
        # Edge transition (optional)
        if self.use_edge_update and z is not None:
            z = self.edge_transition(s, z)
            # Apply mask
            if mask is not None:
                edge_mask = mask[..., None] * mask[..., None, :]
                z = z * edge_mask[..., None]
        
        return s, z, r


class SeqStructureIPA(nn.Module):
    """Simplified sequence-to-structure prediction model.
    
    Based on DyneTrion's IPA architecture but without temporal components.
    Suitable for single structure prediction like AlphaFold2/3.
    """
    
    def __init__(
        self,
        c_s_input: int = 384,       # Input node feature dim (from GeoForm/Protenix)
        c_z_input: int = 128,       # Input edge feature dim
        c_s: int = 256,             # Model node dim
        c_z: int = 128,             # Model edge dim
        num_blocks: int = 4,        # Number of IPA blocks
        use_timestep: bool = True,  # Whether to use diffusion timestep
        timestep_embed_size: int = 64,
        **ipa_kwargs,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.num_blocks = num_blocks
        self.use_timestep = use_timestep
        
        # Input projections
        self.node_expand = nn.Linear(c_s_input, c_s)
        self.edge_expand = nn.Linear(c_z_input, c_z)
        
        # Optional timestep embedding (for diffusion)
        if use_timestep:
            from src.model.utils import get_timestep_embedding
            self.timestep_embed = get_timestep_embedding
            self.node_t_proj = nn.Linear(timestep_embed_size, c_s)
            self.edge_t_proj = nn.Linear(timestep_embed_size, c_z)
        
        # Layer norms on inputs
        self.node_layer_norm = nn.LayerNorm(c_s_input)
        self.edge_layer_norm = nn.LayerNorm(c_z_input)
        
        # IPA blocks
        self.blocks = nn.ModuleList([
            SeqStructureIPABlock(
                c_s=c_s,
                c_z=c_z,
                use_edge_update=(i < num_blocks - 1),  # No edge update on last block
                **ipa_kwargs,
            )
            for i in range(num_blocks)
        ])
        
        # Angle prediction for side chains
        self.angle_resnet = AngleResnet(
            c_s=c_s,
            c_resnet=128,
            no_blocks=2,
            no_angles=7,
            epsilon=1e-12,
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        node_repr: torch.Tensor,    # [B, N, c_s_input] from GeoForm/Protenix
        edge_repr: torch.Tensor,    # [B, N, N, c_z_input]
        rigids_t: torch.Tensor,     # [B, N, 7] initial/noised rigids (quat+trans)
        mask: Optional[torch.Tensor] = None,  # [B, N]
        t: Optional[torch.Tensor] = None,     # [B] diffusion timestep (optional)
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for sequence-to-structure prediction.
        
        Args:
            node_repr: Node features from embedding model
            edge_repr: Edge features from embedding model
            rigids_t: Initial rigids (can be noised for diffusion)
            mask: Residue mask
            t: Diffusion timestep (optional, for diffusion models)
            
        Returns:
            Dictionary with:
                - rigids: [B, N, 7] final rigids
                - angles: [B, N, 7, 2] torsion angles
                - atom37: [B, N, 37, 3] all-atom coordinates
        """
        B, N, _ = node_repr.shape
        
        if mask is None:
            mask = torch.ones(B, N, device=node_repr.device, dtype=torch.float32)
        
        # Expand and layer norm
        node = self.node_layer_norm(node_repr)
        node = self.node_expand(node)  # [B, N, c_s]
        
        edge = self.edge_layer_norm(edge_repr)
        edge = self.edge_expand(edge)  # [B, N, N, c_z]
        
        # Add timestep embedding (optional)
        if self.use_timestep and t is not None:
            t_embed = self.timestep_embed(t, self.node_t_proj.in_features)
            
            # Project and broadcast to all residues
            node_t = self.node_t_proj(t_embed)  # [B, c_s]
            node = node + node_t[:, None, :]    # [B, N, c_s]
            
            edge_t = self.edge_t_proj(t_embed)  # [B, c_z]
            # Broadcast to edges: [B, 1, 1, c_z]
            edge = edge + edge_t[:, None, None, :]
        
        # Initialize rigids from input
        curr_rigids = Rigid.from_tensor_7(rigids_t)
        
        # Store initial for skip connection to angle resnet
        node_initial = node
        
        # Process through IPA blocks
        for block in self.blocks:
            node, edge, curr_rigids = block(node, edge, curr_rigids, mask)
        
        # Predict torsion angles
        unorm_angles, angles = self.angle_resnet(node, node_initial)
        
        # Convert to atom37 coordinates
        atom37 = self._rigids_to_atom37(curr_rigids, angles, mask)
        
        return {
            'rigids': curr_rigids.to_tensor_7(),
            'angles': angles,
            'unorm_angles': unorm_angles,
            'atom37': atom37,
        }
    
    def _rigids_to_atom37(
        self,
        rigids: Rigid,
        angles: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Convert rigids and angles to all-atom coordinates.
        
        Simplified version - just returns backbone atoms for now.
        Full version would use torsion_angles_to_frames -> frames_to_atom37.
        """
        # Get translation component (CA/C4' positions)
        # For now, return backbone positions repeated
        bb_pos = rigids.get_trans()  # [B, N, 3]
        
        # Expand to atom37 format (placeholder)
        atom37 = bb_pos.unsqueeze(-2).expand(*bb_pos.shape[:-1], 37, 3)
        
        # TODO: Implement full all-atom conversion using torsion angles
        # This would use all_atom.torsion_angles_to_frames -> frames_to_atom37_pos
        
        return atom37 * mask[..., None, None]


class SimpleSeqStructureScore(nn.Module):
    """Simplest seq->structure score model (no diffusion timestep).
    
    For direct structure prediction or as a building block.
    """
    
    def __init__(
        self,
        c_s_input: int = 384,
        c_z_input: int = 128,
        c_s: int = 256,
        c_z: int = 128,
        num_blocks: int = 4,
        **ipa_kwargs,
    ):
        super().__init__()
        
        # Just the core IPA model without timestep
        self.ipa_model = SeqStructureIPA(
            c_s_input=c_s_input,
            c_z_input=c_z_input,
            c_s=c_s,
            c_z=c_z,
            num_blocks=num_blocks,
            use_timestep=False,
            **ipa_kwargs,
        )
    
    def forward(
        self,
        node_repr: torch.Tensor,
        edge_repr: torch.Tensor,
        rigids_init: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Direct forward pass without timestep.
        
        Args:
            node_repr: [B, N, c_s_input]
            edge_repr: [B, N, N, c_z_input]
            rigids_init: [B, N, 7] initial rigids (usually identity or from prev iter)
            mask: [B, N]
            
        Returns:
            Output dict with rigids, angles, atom37
        """
        return self.ipa_model(
            node_repr=node_repr,
            edge_repr=edge_repr,
            rigids_t=rigids_init,
            mask=mask,
            t=None,
        )


class DiffusionSeqStructureScore(nn.Module):
    """Seq->structure score model with diffusion timestep.
    
    For diffusion-based structure prediction.
    """
    
    def __init__(
        self,
        c_s_input: int = 384,
        c_z_input: int = 128,
        c_s: int = 256,
        c_z: int = 128,
        num_blocks: int = 4,
        timestep_embed_size: int = 64,
        **ipa_kwargs,
    ):
        super().__init__()
        
        self.ipa_model = SeqStructureIPA(
            c_s_input=c_s_input,
            c_z_input=c_z_input,
            c_s=c_s,
            c_z=c_z,
            num_blocks=num_blocks,
            use_timestep=True,
            timestep_embed_size=timestep_embed_size,
            **ipa_kwargs,
        )
    
    def forward(
        self,
        node_repr: torch.Tensor,
        edge_repr: torch.Tensor,
        rigids_t: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward with diffusion timestep.
        
        Args:
            node_repr: [B, N, c_s_input]
            edge_repr: [B, N, N, c_z_input]
            rigids_t: [B, N, 7] noised rigids
            t: [B] diffusion timestep
            mask: [B, N]
            
        Returns:
            Output dict with predicted structure
        """
        return self.ipa_model(
            node_repr=node_repr,
            edge_repr=edge_repr,
            rigids_t=rigids_t,
            mask=mask,
            t=t,
        )
