"""Minimal Seq -> Structure IPA Model (no heavy OpenFold dependencies).

This version has minimal dependencies and can work without full OpenFold.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Tuple
from openfold.utils.rigid_utils import Rigid, Rotation


class MinimalInvariantPointAttention(nn.Module):
    """Minimal IPA implementation - sufficient for structure prediction."""
    
    def __init__(
        self,
        c_s: int = 256,
        c_z: int = 128,
        c_hidden: int = 16,
        no_heads: int = 8,
        no_qk_points: int = 4,
        no_v_points: int = 8,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        
        # Linear projections
        hc = self.c_hidden * self.no_heads
        self.linear_q = nn.Linear(self.c_s, hc)
        self.linear_kv = nn.Linear(self.c_s, 2 * hc)
        
        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = nn.Linear(self.c_s, hpq)
        
        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = nn.Linear(self.c_s, hpkv)
        
        self.linear_b = nn.Linear(self.c_z, self.no_heads)
        
        self.head_weights = nn.Parameter(torch.zeros(no_heads))
        nn.init.normal_(self.head_weights, mean=0.0, std=0.02)
        
        concat_out_dim = self.no_heads * (self.c_z + self.c_hidden + self.no_v_points * 4)
        self.linear_out = nn.Linear(concat_out_dim, self.c_s)
    
    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        B, N, _ = s.shape
        
        # Scalar attention
        q = self.linear_q(s).reshape(B, N, self.no_heads, self.c_hidden)
        kv = self.linear_kv(s).reshape(B, N, self.no_heads, 2 * self.c_hidden)
        k, v = torch.split(kv, self.c_hidden, dim=-1)
        
        # Point attention
        q_pts = self.linear_q_points(s).reshape(B, N, self.no_heads, self.no_qk_points, 3)
        kv_pts = self.linear_kv_points(s).reshape(B, N, self.no_heads, self.no_qk_points + self.no_v_points, 3)
        k_pts, v_pts = torch.split(kv_pts, [self.no_qk_points, self.no_v_points], dim=-2)
        
        # Apply rigid to points
        q_pts = r[..., None].apply(q_pts.reshape(B, N, self.no_heads * self.no_qk_points, 3))
        q_pts = q_pts.reshape(B, N, self.no_heads, self.no_qk_points, 3)
        
        k_pts = r[..., None].apply(k_pts.reshape(B, N, self.no_heads * self.no_qk_points, 3))
        k_pts = k_pts.reshape(B, N, self.no_heads, self.no_qk_points, 3)
        
        # Compute attention scores
        # q, k: [B, N, H, C] -> [B, H, N, C]
        q_perm = q.permute(0, 2, 1, 3)  # [B, H, N, C]
        k_perm = k.permute(0, 2, 1, 3)  # [B, H, N, C]
        a = torch.matmul(q_perm, k_perm.transpose(-2, -1))  # [B, H, N, N]
        a = a * math.sqrt(1.0 / (3 * self.c_hidden))
        
        if z is not None:
            b = self.linear_b(z)  # [B, N, N, H]
            a = a + math.sqrt(1.0 / 3) * b.permute(0, 3, 1, 2)  # [B, H, N, N]
        
        # Point attention
        # q_pts: [B, N, H, P_qk, 3], k_pts: [B, N, H, P_qk, 3]
        q_pts_sq = (q_pts ** 2).sum(dim=-1)  # [B, N, H, P_qk]
        k_pts_sq = (k_pts ** 2).sum(dim=-1)  # [B, N, H, P_qk]
        
        # Compute pairwise distances between points
        # Result: [B, H, N, N]
        q_pts_flat = q_pts.permute(0, 2, 1, 3, 4).reshape(B, self.no_heads, N, -1)  # [B, H, N, P_qk*3]
        k_pts_flat = k_pts.permute(0, 2, 1, 3, 4).reshape(B, self.no_heads, N, -1)  # [B, H, N, P_qk*3]
        
        # Compute distance matrix: ||q - k||^2 = ||q||^2 + ||k||^2 - 2*q.k
        q_sq = q_pts_sq.sum(dim=-1).permute(0, 2, 1)  # [B, H, N]
        k_sq = k_pts_sq.sum(dim=-1).permute(0, 2, 1)  # [B, H, N]
        
        dot_prod = torch.matmul(q_pts_flat, k_pts_flat.transpose(-2, -1))  # [B, H, N, N]
        
        pt_att = q_sq[..., :, None] + k_sq[..., None, :] - 2 * dot_prod  # [B, H, N, N]
        pt_att = pt_att * math.sqrt(1.0 / (3 * self.no_qk_points * 3))  # Normalize by feature dim
        
        a = a + pt_att
        
        # Mask
        if mask is not None:
            mask_bias = (1e5 * (mask - 1))[..., None, None, :]
            a = a + mask_bias
        
        a = torch.softmax(a, dim=-1)
        
        # Output
        o = torch.matmul(a, v.transpose(-3, -2)).transpose(-3, -2)
        o = o.reshape(B, N, -1)
        
        # Point output
        # v_pts: [B, N, H, V, 3]
        # a: [B, H, N, N]
        # Output: [B, N, H, V, 3]
        v_pts_perm = v_pts.permute(0, 2, 1, 3, 4)  # [B, H, N, V, 3]
        o_pt = torch.einsum('bhmn,bhmvd->bhnvd', a, v_pts_perm)  # [B, H, N, V, 3]
        o_pt = o_pt.permute(0, 2, 1, 3, 4)  # [B, N, H, V, 3]
        o_pt = o_pt.reshape(B, N, -1, 3)  # [B, N, H*V, 3]
        
        o_pt_norm = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + 1e-8)  # [B, N, H*V]
        o_pt_norm = o_pt_norm.reshape(B, N, -1)
        
        # Concatenate outputs
        o = o.reshape(B, N, -1)  # [B, N, H*C]
        if z is not None:
            # z: [B, N, N, C_z]
            # a: [B, H, N, N]
            o_pair = torch.einsum('bhij,bijc->bihc', a, z)  # [B, N, H, C_z]
            out = torch.cat([o, o_pt.reshape(B, N, -1), o_pt_norm, o_pair.reshape(B, N, -1)], dim=-1)
        else:
            out = torch.cat([o, o_pt.reshape(B, N, -1), o_pt_norm], dim=-1)
        
        out = self.linear_out(out)
        return out


class MinimalStructureModuleTransition(nn.Module):
    """Simple transition layer for structure module."""
    
    def __init__(self, c: int, n: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(c),
                nn.Linear(c, c * 2),
                nn.ReLU(),
                nn.Linear(c * 2, c),
            )
            for i in range(n)
        ])
    
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            s = s + layer(s)
        return s


class MinimalBackboneUpdate(nn.Module):
    """Predicts rigid updates from node features."""
    
    def __init__(self, c_s: int):
        super().__init__()
        self.linear = nn.Linear(c_s, 6)  # 3 for rotation, 3 for translation
    
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(s)


class MinimalEdgeTransition(nn.Module):
    """Updates edge features from node features."""
    
    def __init__(self, c_s: int, c_z: int, n: int = 2):
        super().__init__()
        self.c_z = c_z
        
        # Input is c_z + 2*c_s (z concatenated with s_i and s_j)
        input_dim = c_z + 2 * c_s
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, c_z),
                nn.ReLU(),
            )
        ])
        
        # Subsequent layers take c_z as input
        for _ in range(n - 1):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(c_z),
                nn.Linear(c_z, c_z),
                nn.ReLU(),
            ))
    
    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = s.shape
        
        # Concatenate node features to edges
        s_i = s[..., None, :].expand(B, N, N, -1)
        s_j = s[..., None, :, :].expand(B, N, N, -1)
        
        z = torch.cat([z, s_i, s_j], dim=-1)
        
        for i, layer in enumerate(self.layers):
            if i == 0:
                z = layer(z)  # First layer projects to c_z
            else:
                z = z + layer(z)  # Residual for subsequent layers
        
        return z


class MinimalAngleResnet(nn.Module):
    """Predicts torsion angles."""
    
    def __init__(
        self,
        c_s: int = 256,
        c_resnet: int = 128,
        no_blocks: int = 2,
        no_angles: int = 7,
    ):
        super().__init__()
        self.c_s = c_s
        self.no_angles = no_angles
        
        self.initial_proj = nn.Linear(c_s * 2, c_resnet)
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c_resnet, c_resnet),
                nn.ReLU(),
                nn.Linear(c_resnet, c_resnet),
                nn.ReLU(),
            )
            for _ in range(no_blocks)
        ])
        
        self.final_proj = nn.Linear(c_resnet, no_angles * 2)
    
    def forward(
        self,
        s: torch.Tensor,
        s_initial: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns unnormalized angles and normalized angles (sin/cos)."""
        s_cat = torch.cat([s, s_initial], dim=-1)
        x = self.initial_proj(s_cat)
        
        for block in self.blocks:
            x = x + block(x)
        
        unorm_angles = self.final_proj(x).reshape(*x.shape[:-1], self.no_angles, 2)
        
        # Normalize to sin/cos
        angles = unorm_angles / (torch.norm(unorm_angles, dim=-1, keepdim=True) + 1e-8)
        
        return unorm_angles, angles


class SeqStructureIPASimplified(nn.Module):
    """Simplified Seq->Structure model with minimal dependencies."""
    
    def __init__(
        self,
        c_s_input: int = 384,
        c_z_input: int = 128,
        c_s: int = 256,
        c_z: int = 128,
        num_blocks: int = 4,
        use_timestep: bool = False,
        timestep_embed_size: int = 64,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.num_blocks = num_blocks
        self.use_timestep = use_timestep
        
        # Input projections
        self.node_expand = nn.Linear(c_s_input, c_s)
        self.edge_expand = nn.Linear(c_z_input, c_z)
        
        # Optional timestep
        if use_timestep:
            self.node_t_proj = nn.Linear(timestep_embed_size, c_s)
            self.edge_t_proj = nn.Linear(timestep_embed_size, c_z)
        
        self.node_ln = nn.LayerNorm(c_s_input)
        self.edge_ln = nn.LayerNorm(c_z_input)
        
        # IPA blocks
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ipa': MinimalInvariantPointAttention(c_s, c_z),
                'ipa_dropout': nn.Dropout(0.0),
                'ipa_ln': nn.LayerNorm(c_s),
                'transition': MinimalStructureModuleTransition(c_s),
                'bb_update': MinimalBackboneUpdate(c_s),
                'edge_transition': MinimalEdgeTransition(c_s, c_z) if i < num_blocks - 1 else None,
            })
            for i in range(num_blocks)
        ])
        
        self.angle_resnet = MinimalAngleResnet(c_s)
    
    def forward(
        self,
        node_repr: torch.Tensor,
        edge_repr: torch.Tensor,
        rigids_t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        B, N, _ = node_repr.shape
        
        if mask is None:
            mask = torch.ones(B, N, device=node_repr.device)
        
        # Expand inputs
        node = self.node_ln(node_repr)
        node = self.node_expand(node)
        
        edge = self.edge_ln(edge_repr)
        edge = self.edge_expand(edge)
        
        # Add timestep
        if self.use_timestep and t is not None:
            # Simple sinusoidal embedding
            t_embed = self._timestep_embedding(t, 64)
            node = node + self.node_t_proj(t_embed)[:, None, :]
            edge = edge + self.edge_t_proj(t_embed)[:, None, None, :]
        
        # Initialize rigids
        curr_rigids = Rigid.from_tensor_7(rigids_t)
        node_initial = node
        
        # IPA blocks
        for block in self.blocks:
            # IPA
            ipa_out = block['ipa'](node, edge, curr_rigids, mask)
            ipa_out = block['ipa_dropout'](ipa_out)
            node = block['ipa_ln'](node + ipa_out)
            
            # Transition
            node = block['transition'](node)
            
            # Backbone update
            rigid_update = block['bb_update'](node)
            curr_rigids = curr_rigids.compose_q_update_vec(rigid_update)
            
            # Edge transition
            if block['edge_transition'] is not None:
                edge = block['edge_transition'](node, edge)
                edge_mask = mask[..., None] * mask[..., None, :]
                edge = edge * edge_mask[..., None]
        
        # Angles
        unorm_angles, angles = self.angle_resnet(node, node_initial)
        
        # Simple atom37 (backbone only for now)
        bb_pos = curr_rigids.get_trans()
        atom37 = bb_pos.unsqueeze(-2).expand(*bb_pos.shape[:-1], 37, 3)
        
        return {
            'rigids': curr_rigids.to_tensor_7(),
            'angles': angles,
            'unorm_angles': unorm_angles,
            'atom37': atom37 * mask[..., None, None],
        }
    
    def _timestep_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal timestep embedding."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# Convenience classes
class SimpleSeqStructureScore(SeqStructureIPASimplified):
    """Simple version without timestep."""
    
    def __init__(self, **kwargs):
        super().__init__(use_timestep=False, **kwargs)


class DiffusionSeqStructureScore(SeqStructureIPASimplified):
    """Diffusion version with timestep."""
    
    def __init__(self, **kwargs):
        super().__init__(use_timestep=True, **kwargs)
