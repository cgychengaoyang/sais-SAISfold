"""Score-based IPA model for protein structure diffusion.

This module implements a SE(3) equivariant model for protein structure prediction
using Invariant Point Attention (IPA) blocks and score-based diffusion training.

Following DyneTrion's approach, scores are computed from the predicted structure
relative to the initial noisy structure, rather than directly predicted from features.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from src.data import all_atom
from openfold.model.structure_module import AngleResnet

# Use checkpoint if available
try:
    from torch.utils.checkpoint import checkpoint as cp
    HAS_CHECKPOINT = True
except ImportError:
    HAS_CHECKPOINT = False

class Rigid:
    """Minimal Rigid class for backbone frames."""
    
    def __init__(self, rots, trans):
        """
        Args:
            rots: Rotation matrices [..., 3, 3] or quaternions [..., 4]
            trans: Translation vectors [..., 3]
        """
        if rots.shape[-1] == 4:
            # Convert quat to rotation matrix
            self._rots = self._quat_to_rot(rots)
            self._quats = rots
        else:
            self._rots = rots
            self._quats = self._rot_to_quat(rots)
        self._trans = trans
    
    @staticmethod
    def _quat_to_rot(quat):
        """Convert quaternion to rotation matrix."""
        # Normalize quaternion
        quat = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)
        w, x, y, z = quat.unbind(dim=-1)
        
        # Flatten batch dims
        orig_shape = quat.shape[:-1]
        quat = quat.reshape(-1, 4)
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Build rotation matrix
        rot = torch.stack([
            1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x),
            2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)
        ], dim=-1).reshape(-1, 3, 3)
        
        # Reshape back
        rot = rot.reshape(*orig_shape, 3, 3)
        return rot
    
    @staticmethod
    def _rot_to_quat(rot):
        """Convert rotation matrix to quaternion."""
        # Flatten batch dims
        orig_shape = rot.shape[:-2]
        rot = rot.reshape(-1, 3, 3)
        
        # Compute quaternion from rotation matrix
        trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
        
        w = torch.sqrt(1 + trace + 1e-8) / 2
        x = (rot[:, 2, 1] - rot[:, 1, 2]) / (4 * w + 1e-8)
        y = (rot[:, 0, 2] - rot[:, 2, 0]) / (4 * w + 1e-8)
        z = (rot[:, 1, 0] - rot[:, 0, 1]) / (4 * w + 1e-8)
        
        quat = torch.stack([w, x, y, z], dim=-1)
        quat = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Reshape back
        quat = quat.reshape(*orig_shape, 4)
        return quat
    
    @classmethod
    def from_tensor_7(cls, tensor_7):
        """Create from [..., 7] tensor (quat [4] + trans [3])."""
        quat = tensor_7[..., :4]
        trans = tensor_7[..., 4:]
        return cls(quat, trans)
    
    def to_tensor_7(self):
        """Convert to [..., 7] tensor (quat [4] + trans [3])."""
        return torch.cat([self._quats, self._trans], dim=-1)
    
    def get_rots(self):
        """Get rotation matrices."""
        return self._rots
    
    def get_quats(self):
        """Get quaternions."""
        return self._quats
    
    def get_trans(self):
        """Get translations."""
        return self._trans
    
    def apply(self, points):
        """Apply rigid transformation to points.
        
        Args:
            points: [B, N, ..., 3] where B,N match the Rigid batch dims
            
        Returns:
            Transformed points with same shape as input
        """
        # points: [B, N, ..., 3]
        # rots: [B, N, 3, 3]
        # trans: [B, N, 3]
        B, N = points.shape[0], points.shape[1]
        
        # Get the extra dimensions after B,N
        extra_dims = points.shape[2:-1]  # e.g., (heads*points,) or (heads, points)
        num_extra = int(torch.prod(torch.tensor(extra_dims))) if extra_dims else 1
        
        # Reshape points to [B, N, num_extra, 3]
        points_reshaped = points.reshape(B, N, num_extra, 3)
        
        # Expand rotation: [B, N, 3, 3] -> [B, N, 1, 3, 3] -> [B, N, num_extra, 3, 3]
        rots_expanded = self._rots.unsqueeze(2).expand(-1, -1, num_extra, -1, -1)
        
        # Apply rotation: [B, N, num_extra, 3, 3] @ [B, N, num_extra, 3, 1]
        rotated = torch.einsum('bnhij,bnhj->bnhi', rots_expanded, points_reshaped)
        
        # Add translation: [B, N, 3] -> [B, N, 1, 3]
        trans_expanded = self._trans.unsqueeze(2)
        result = rotated + trans_expanded
        
        # Reshape back to original shape
        return result.reshape(points.shape)
    
    def apply_inverse(self, points):
        """Apply inverse rigid transformation to points.
        
        Args:
            points: [B, N, ..., 3] where B,N match the Rigid batch dims
            
        Returns:
            Inverse transformed points with same shape as input
        """
        B, N = points.shape[0], points.shape[1]
        
        # Get extra dimensions
        extra_dims = points.shape[2:-1]
        num_extra = int(torch.prod(torch.tensor(extra_dims))) if extra_dims else 1
        
        # Reshape points
        points_reshaped = points.reshape(B, N, num_extra, 3)
        
        # Subtract translation first
        trans_expanded = self._trans.unsqueeze(2)  # [B, N, 1, 3]
        local_points = points_reshaped - trans_expanded  # [B, N, num_extra, 3]
        
        # Apply inverse rotation
        inv_rots = self._rots.transpose(-2, -1)  # [B, N, 3, 3]
        inv_rots_expanded = inv_rots.unsqueeze(2).expand(-1, -1, num_extra, -1, -1)
        
        result = torch.einsum('bnhij,bnhj->bnhi', inv_rots_expanded, local_points)
        
        return result.reshape(points.shape)
    
    def invert(self):
        """Invert the rigid transformation."""
        inv_rot = self._rots.transpose(-2, -1)
        inv_trans = -torch.einsum('...ij,...j->...i', inv_rot, self._trans)
        return Rigid(inv_rot, inv_trans)
    
    def compose(self, other):
        """Compose with another rigid: self(other(x))."""
        new_rot = torch.einsum('...ij,...jk->...ik', self._rots, other._rots)
        new_trans = torch.einsum('...ij,...j->...i', self._rots, other._trans) + self._trans
        return Rigid(new_rot, new_trans)
    
    def compose_q_update_vec(self, update):
        """Compose with an update vector [..., 6] (rotvec [3] + trans [3])."""
        # Split update into rotation and translation
        rotvec = update[..., :3]
        trans_update = update[..., 3:]
        
        # Clamp values to prevent numerical instability
        rotvec = torch.clamp(rotvec, -10, 10)
        trans_update = torch.clamp(trans_update, -100, 100)
        
        # Convert rotvec to rotation matrix
        angle = rotvec.norm(dim=-1, keepdim=True) + 1e-8
        axis = rotvec / angle
        
        # Rodrigues formula
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        # Build skew-symmetric matrix
        K = torch.zeros(*axis.shape[:-1], 3, 3, device=axis.device, dtype=axis.dtype)
        K[..., 0, 1] = -axis[..., 2]
        K[..., 0, 2] = axis[..., 1]
        K[..., 1, 0] = axis[..., 2]
        K[..., 1, 2] = -axis[..., 0]
        K[..., 2, 0] = -axis[..., 1]
        K[..., 2, 1] = axis[..., 0]
        
        # Rotation matrix
        I = torch.eye(3, device=axis.device, dtype=axis.dtype)
        R_update = I + sin_a[..., None] * K + (1 - cos_a[..., None]) * torch.einsum('...ij,...jk->...ik', K, K)
        
        # Compose rotations
        new_rot = torch.einsum('...ij,...jk->...ik', R_update, self._rots)
        
        # Update translation
        new_trans = self._trans + trans_update
        
        return Rigid(new_rot, new_trans)


def quat_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.stack([w, x, y, z], dim=-1)


def quat_to_rotvec(quat):
    """Convert quaternion to rotation vector (axis-angle representation).
    
    Numerically stable version that avoids NaN gradients in backward pass.
    """
    # Normalize quaternion
    quat = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)
    
    w = quat[..., 0]
    xyz = quat[..., 1:]
    
    # Clamp w to avoid gradient issues at boundaries
    # Use a slightly smaller range to avoid exact ±1.0 values
    w_clamped = torch.clamp(w, -0.99999, 0.99999)
    
    # Compute angle
    angle = 2 * torch.acos(w_clamped)
    
    # Compute axis with better numerical stability
    # Use the identity: sin^2(θ/2) = (1 - cos(θ))/2 = (1 - w)/2 for unit quat
    # sin(θ/2) = sqrt((1 - w) * (1 + w)) for w near ±1
    sin_sq = (1.0 - w_clamped) * (1.0 + w_clamped)
    sin_half_angle = torch.sqrt(torch.clamp(sin_sq, min=1e-10))
    
    # Safe division with mask for small angles
    axis = xyz / (sin_half_angle[..., None] + 1e-8)
    
    # Rotation vector
    rotvec = axis * angle[..., None]
    
    # For very small rotations (w close to 1), use linear approximation
    # This avoids numerical issues when angle ≈ 0
    small_angle_mask = (w > 0.999).unsqueeze(-1)
    # When angle is small: rotvec ≈ 2 * xyz (since sin(θ/2) ≈ θ/2 and angle ≈ 2*sin(θ/2))
    rotvec_linear = 2.0 * xyz
    rotvec = torch.where(small_angle_mask, rotvec_linear, rotvec)
    
    return rotvec


def rotvec_to_quat(rotvec):
    """Convert rotation vector to quaternion."""
    angle = rotvec.norm(dim=-1, keepdim=True)
    axis = rotvec / (angle + 1e-8)
    
    half_angle = angle / 2
    w = torch.cos(half_angle)
    xyz = axis * torch.sin(half_angle)
    
    return torch.cat([w, xyz], dim=-1)


class IPABlock(nn.Module):
    """Invariant Point Attention block."""
    
    def __init__(
        self,
        c_s: int = 256,
        c_z: int = 128,
        c_hidden: int = 16,
        no_heads: int = 8,
        no_qk_points: int = 4,
        no_v_points: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        
        # Scalar attention
        self.linear_q = nn.Linear(c_s, c_hidden * no_heads, bias=False)
        self.linear_kv = nn.Linear(c_s, 2 * c_hidden * no_heads, bias=False)
        self.linear_z = nn.Linear(c_z, no_heads, bias=False)
        self.linear_o = nn.Linear(c_hidden * no_heads, c_s)
        
        # Point attention
        self.linear_q_points = nn.Linear(c_s, no_qk_points * no_heads * 3, bias=False)
        self.linear_kv_points = nn.Linear(c_s, (no_qk_points + no_v_points) * no_heads * 3, bias=False)
        self.linear_o_points = nn.Linear(no_v_points * no_heads * 3, c_s)
        
        # Gating
        self.gate = nn.Linear(c_s, no_heads)
        
        # Layer norms
        self.layer_norm_s = nn.LayerNorm(c_s)
        self.layer_norm_z = nn.LayerNorm(c_z)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.zeros_(self.linear_o.weight)
        nn.init.zeros_(self.linear_o.bias)
        nn.init.zeros_(self.linear_o_points.weight)
        nn.init.zeros_(self.linear_o_points.bias)
    
    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        rigids: Rigid,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            s: [B, N, C_s] single representation
            z: [B, N, N, C_z] pair representation
            rigids: Rigid objects [B, N]
            mask: [B, N] optional mask
            
        Returns:
            [B, N, C_s] updated single representation
        """
        B, N, _ = s.shape
        
        # Pre-normalization
        s = self.layer_norm_s(s)
        z = self.layer_norm_z(z)
        
        # Compute queries and keys/values
        q = self.linear_q(s).reshape(B, N, self.no_heads, self.c_hidden)
        kv = self.linear_kv(s).reshape(B, N, self.no_heads, 2 * self.c_hidden)
        k, v = kv.chunk(2, dim=-1)
        
        # Compute point queries and keys/values
        q_points = self.linear_q_points(s).reshape(B, N, self.no_heads, self.no_qk_points, 3)
        kv_points = self.linear_kv_points(s).reshape(B, N, self.no_heads, self.no_qk_points + self.no_v_points, 3)
        k_points, v_points = kv_points.split([self.no_qk_points, self.no_v_points], dim=-2)
        
        # Rotate points into local frames
        # [B, N, heads, points, 3]
        q_points_local = rigids.apply_inverse(q_points.reshape(B, N, -1, 3)).reshape(B, N, self.no_heads, self.no_qk_points, 3)
        k_points_local = rigids.apply_inverse(k_points.reshape(B, N, -1, 3)).reshape(B, N, self.no_heads, self.no_qk_points, 3)
        v_points_local = rigids.apply_inverse(v_points.reshape(B, N, -1, 3)).reshape(B, N, self.no_heads, self.no_v_points, 3)
        
        # Attention logits from scalar attention
        attn = torch.einsum('bnhd,bmhd->bhnm', q, k) / math.sqrt(self.c_hidden)
        
        # Add pair bias
        attn = attn + self.linear_z(z).permute(0, 3, 1, 2)
        
        # Add point attention term (squared distances)
        # q_points_local: [B, N, heads, qk_points, 3], k_points_local: [B, N, heads, qk_points, 3]
        # We need to compute distance between each query position i and key position j
        qk_diff = q_points_local[:, :, None] - k_points_local[:, None]  # [B, N, N, heads, qk_points, 3]
        point_dist = qk_diff.norm(dim=-1).mean(dim=-1)  # [B, N, N, heads]
        # Permute to match attn shape [B, heads, N, N]
        point_dist = point_dist.permute(0, 3, 1, 2)  # [B, heads, N, N]
        attn = attn - point_dist * 0.5  # Scale factor
        
        # Apply mask
        if mask is not None:
            attn_mask = mask[:, None, None, :] * mask[:, None, :, None]
            attn = attn.masked_fill(~attn_mask.bool(), -1e9)
        
        # Softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply gating
        gate = torch.sigmoid(self.gate(s))
        attn = attn * gate.permute(0, 2, 1).unsqueeze(-1)
        
        # Aggregate values
        o = torch.einsum('bhnm,bmhd->bnhd', attn, v)
        o = o.reshape(B, N, -1)
        o = self.linear_o(o)
        
        # Aggregate point values
        o_points = torch.einsum('bhnm,bmhpd->bnhpd', attn, v_points_local)
        o_points = o_points.reshape(B, N, -1, 3)
        o_points = rigids.apply(o_points.reshape(B, N, -1, 3)).reshape(B, N, self.no_heads * self.no_v_points * 3)
        o_points = self.linear_o_points(o_points)
        
        return o + o_points


class Transition(nn.Module):
    """Transition layer for node features with residual connection.
    
    Following SE3 Diffusion / DyneTrion: 3-layer with residual, post-LN
    """
    
    def __init__(self, c_s: int, expansion_factor: int = 4, dropout: float = 0.0):
        super().__init__()
        self.c_s = c_s
        self.expansion_factor = expansion_factor
        
        # 3-layer MLP like SE3 Diffusion / DyneTrion
        self.linear_1 = nn.Linear(c_s, expansion_factor * c_s)
        self.linear_2 = nn.Linear(expansion_factor * c_s, expansion_factor * c_s)
        self.linear_3 = nn.Linear(expansion_factor * c_s, c_s)
        self.dropout = nn.Dropout(dropout)
        
        # Post-layer norm (like references)
        self.layer_norm = nn.LayerNorm(c_s)
        
        # Initialize last layer to zeros (final_init)
        nn.init.zeros_(self.linear_3.weight)
        nn.init.zeros_(self.linear_3.bias)
    
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: [B, N, C_s]
            
        Returns:
            [B, N, C_s]
        """
        s_initial = s
        s = self.linear_1(s)
        s = F.relu(s)
        s = self.linear_2(s)
        s = F.relu(s)
        s = self.dropout(s)
        s = self.linear_3(s)
        s = s + s_initial  # Residual connection
        s = self.layer_norm(s)  # Post-LN
        return s


class EdgeTransition(nn.Module):
    """Transition layer for edge features."""
    
    def __init__(self, c_s: int, c_z: int, n: int = 4, dropout: float = 0.0):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        
        self.layer_norm = nn.LayerNorm(c_z)
        
        # Outer product mean
        self.linear_s_i = nn.Linear(c_s, n)
        self.linear_s_j = nn.Linear(c_s, n)
        
        self.linear_out = nn.Sequential(
            nn.Linear(c_z + n * n, c_z),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(c_z, c_z),
        )
    
    def forward(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: [B, N, C_s]
            z: [B, N, N, C_z]
            
        Returns:
            [B, N, N, C_z]
        """
        z = self.layer_norm(z)
        
        # Outer product mean
        a = self.linear_s_i(s)
        b = self.linear_s_j(s)
        outer = torch.einsum('bni,bnj->bnij', a, b)
        outer = outer.reshape(*outer.shape[:2], -1)
        
        # Broadcast and concatenate
        outer = outer.unsqueeze(2).expand(-1, -1, z.shape[2], -1)
        
        z = torch.cat([z, outer], dim=-1)
        z = self.linear_out(z)
        
        return z


class BackboneUpdate(nn.Module):
    """Predicts rigid backbone updates.
    
    Following SE3 Diffusion / DyneTrion: zero initialization (init="final")
    This ensures identity transformation at initialization.
    """
    
    def __init__(self, c_s: int):
        super().__init__()
        self.linear = nn.Linear(c_s, 6)
        
        # Zero initialization (final_init) - identity transformation at start
        # This is crucial for stable training
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: [B, N, C_s]
            
        Returns:
            [B, N, 6] rigid update (rotvec [3] + trans [3])
        """
        update = self.linear(s)
        # Clamp to prevent extreme values that cause NaN in compose_q_update_vec
        update = torch.clamp(update, -10, 10)
        return update


class DyneTrionScoreNet(nn.Module):
    """DyneTrion-inspired score network for protein structure diffusion.
    
    This model predicts the score (gradient of log probability) for protein
    structure denoising, which is used in the diffusion model.
    
    Key improvements from SE3 Diffusion (Jakkola lab):
    - Coordinate scaling for stable IPA point attention
    - Direct score prediction from node features
    """
    
    def __init__(
        self,
        c_s_input: int = 384,
        c_z_input: int = 128,
        c_s: int = 256,
        c_z: int = 128,
        num_blocks: int = 4,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
        ipa_config: Optional[Dict] = None,
        coordinate_scaling: float = 0.1,  # Key for stable training
    ):
        super().__init__()
        self.c_s_input = c_s_input
        self.c_z_input = c_z_input
        self.c_s = c_s
        self.coordinate_scaling = coordinate_scaling
        self.c_z = c_z
        self.num_blocks = num_blocks
        self.use_checkpoint = use_checkpoint and HAS_CHECKPOINT
        self.diffuser = None  # Set via set_diffuser()
        
        # Input projections
        self.node_proj = nn.Sequential(
            nn.LayerNorm(c_s_input),
            nn.Linear(c_s_input, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )
        
        self.edge_proj = nn.Sequential(
            nn.LayerNorm(c_z_input),
            nn.Linear(c_z_input, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
        )
        
        # Timestep projection (for diffusion conditioning)
        self.t_proj = nn.Linear(c_s + 1, c_s)
        
        # IPA blocks
        ipa_kwargs = {
            'c_s': c_s,
            'c_z': c_z,
            'c_hidden': 256,
            'no_heads': 8,
            'no_qk_points': 8,
            'no_v_points': 12,
            'dropout': dropout,
        }
        if ipa_config:
            ipa_kwargs.update(ipa_config)
        
        self.trunk = nn.ModuleDict()
        for b in range(num_blocks):
            self.trunk[f'ipa_{b}'] = IPABlock(**ipa_kwargs)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(c_s)
            self.trunk[f'node_transition_{b}'] = Transition(c_s, dropout=dropout)
            self.trunk[f'bb_update_{b}'] = BackboneUpdate(c_s)
            
            # Sequence transformer with skip connection (SE3 Diffusion / DyneTrion)
            self.trunk[f'skip_embed_{b}'] = nn.Linear(c_s, c_s)
            tfmr_layer = nn.TransformerEncoderLayer(
                d_model=2 * c_s,
                nhead=8,
                dim_feedforward=2 * c_s,
                dropout=dropout,
                batch_first=False,
            )
            self.trunk[f'seq_tfmr_{b}'] = nn.TransformerEncoder(tfmr_layer, num_layers=1)
            self.trunk[f'post_tfmr_{b}'] = nn.Linear(2 * c_s, c_s)
            
            if b < num_blocks - 1:
                self.trunk[f'edge_transition_{b}'] = EdgeTransition(c_s, c_z, dropout=dropout)
        
        # Torsion angle prediction head (DyneTrion-style)
        self.angle_resnet = AngleResnet(
            c_in=c_s,
            c_hidden=c_s,
            no_blocks=2,
            no_angles=7,
            epsilon=1e-12,
        )
    
    def _apply_trans_fn(self, rigids: Rigid, trans_fn) -> Rigid:
        """Apply a transformation function to the translation component of rigids."""
        new_trans = trans_fn(rigids.get_trans())
        return Rigid(rigids.get_rots(), new_trans)
    
    def _forward_block(
        self,
        b: int,
        node: torch.Tensor,
        edge: torch.Tensor,
        curr_rigids: Rigid,
        mask: Optional[torch.Tensor],
        apply_backbone_update: bool = False,
    ) -> Tuple[torch.Tensor, Rigid, torch.Tensor]:
        """Forward pass for a single IPA block."""
        # IPA
        ipa_embed = self.trunk[f'ipa_{b}'](node, edge, curr_rigids, mask)
        node = self.trunk[f'ipa_ln_{b}'](node + ipa_embed)
        
        # Transition
        node = self.trunk[f'node_transition_{b}'](node)
        
        # Backbone update (always applied to get predicted structure)
        rigid_update = self.trunk[f'bb_update_{b}'](node)
        if mask is not None:
            rigid_update = rigid_update * mask[..., None]
        
        # Apply update using compose_q_update_vec (DyneTrion-style)
        curr_rigids = curr_rigids.compose_q_update_vec(rigid_update)
        
        # Edge transition (except last block)
        if b < self.num_blocks - 1:
            edge = self.trunk[f'edge_transition_{b}'](node, edge)
        
        return node, curr_rigids, edge
    
    def set_diffuser(self, diffuser):
        """Set the diffuser for proper score computation.
        
        Args:
            diffuser: SE3Diffuser instance
        """
        self.diffuser = diffuser
    
    def _compute_scores_from_rigids(
        self,
        rigids_0: Rigid,
        rigids_t: Rigid,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rotation and translation scores from predicted vs initial rigids.
        
        Following SE3 Diffusion and DyneTrion approach:
        - Rotation score is computed using the diffuser's calc_rot_score method
        - Translation score is computed using the diffuser's calc_trans_score method
        
        Args:
            rigids_0: Predicted clean rigids (from model output)
            rigids_t: Initial noisy rigids (input to model)
            t: Diffusion timestep [B]
            
        Returns:
            rot_score: [B, N, 3] rotation score (rotation vector)
            trans_score: [B, N, 3] translation score
        """
        # If diffuser is available, use it for proper score computation
        if self.diffuser is not None:
            # Import OpenFold's Rigid class for the diffuser
            from openfold.utils import rigid_utils as ru
            
            # Convert local Rigid to OpenFold Rigid
            # OpenFold Rigid.from_tensor_7 expects [B, N, 7] (quat [4] + trans [3])
            rigids_0_tensor = rigids_0.to_tensor_7()  # [B, N, 7]
            rigids_t_tensor = rigids_t.to_tensor_7()  # [B, N, 7]
            
            ru_rigids_0 = ru.Rigid.from_tensor_7(rigids_0_tensor)
            ru_rigids_t = ru.Rigid.from_tensor_7(rigids_t_tensor)
            
            # Use diffuser's methods (like SE3 Diffusion/DyneTrion)
            rot_score = self.diffuser.calc_rot_score(
                ru_rigids_t,
                ru_rigids_0,
                t
            )
            trans_score = self.diffuser.calc_trans_score(
                ru_rigids_t.get_trans(),
                ru_rigids_0.get_trans(),
                t[:, None, None] if t.dim() == 1 else t,
                use_torch=True
            )
        else:
            # Fallback: compute simple scores without IGSO(3) scaling
            # This is less accurate but works without a diffuser
            
            # Compute relative rotation: quats_t^{-1} * quats_0 (points from t to 0)
            quats_t_inv = rigids_t.invert().get_quats()
            quats_0 = rigids_0.get_quats()
            quats_rel = quat_multiply(quats_t_inv, quats_0)  # [B, N, 4]
            
            # Convert to rotation vector (axis-angle)
            rot_score = quat_to_rotvec(quats_rel)  # [B, N, 3]
            
            # Translation score: difference from noisy to clean (points from t to 0)
            trans_0 = rigids_0.get_trans()  # [B, N, 3]
            trans_t = rigids_t.get_trans()  # [B, N, 3]
            trans_score = trans_0 - trans_t  # [B, N, 3]  # FIXED: was inverted!
        
        return rot_score, trans_score
    
    def forward(
        self,
        node_repr: torch.Tensor,
        edge_repr: torch.Tensor,
        rigids: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        apply_backbone_updates: bool = True,  # Changed default to True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with structure refinement and score prediction.
        
        Args:
            node_repr: [B, N, C_s_input] sequence/node embeddings
            edge_repr: [B, N, N, C_z_input] pair/edge embeddings
            rigids: [B, N, 7] initial rigids (quat [4] + trans [3])
            t: [B] diffusion timestep
            mask: [B, N] optional mask
            apply_backbone_updates: Whether to apply backbone updates (kept for API compatibility)
            
        Returns:
            Dictionary with:
                - rigids: [B, N, 7] refined rigids (predicted clean structure)
                - pred_rot_score: [B, N, 3] predicted rotation score
                - pred_trans_score: [B, N, 3] predicted translation score
                - node_repr: [B, N, C_s] final node features
                - edge_repr: [B, N, N, C_z] final edge features
        """
        B, N = node_repr.shape[:2]
        
        # Initialize rigids
        rigids_tensor = rigids.clone()
        rigids_tensor[..., :4] = rigids_tensor[..., :4] / (rigids_tensor[..., :4].norm(dim=-1, keepdim=True) + 1e-8)
        curr_rigids = Rigid.from_tensor_7(rigids_tensor)
        init_rigids = Rigid.from_tensor_7(rigids_tensor)
        
        # Apply coordinate scaling for stable IPA (SE3 Diffusion best practice)
        curr_rigids = self._apply_trans_fn(curr_rigids, lambda x: x * self.coordinate_scaling)
        
        # Project inputs
        node = self.node_proj(node_repr)
        edge = self.edge_proj(edge_repr)
        
        # Add timestep embedding to node features
        t_expanded = t.view(B, 1, 1).expand(B, N, 1)  # [B, N, 1]
        node_with_t = torch.cat([node, t_expanded], dim=-1)  # [B, N, C_s+1]
        node = self.t_proj(node_with_t)
        
        if mask is not None:
            node = node * mask[..., None]
            edge_mask = mask[..., None] * mask[..., None, :]
            edge = edge * edge_mask[..., None]
        
        # Save initial node for skip connections
        init_node = node
        
        # IPA blocks with sequence transformers (SE3 Diffusion / DyneTrion style)
        for b in range(self.num_blocks):
            node, curr_rigids, edge = self._forward_block(
                b, node, edge, curr_rigids, mask, apply_backbone_update=True
            )
            
            # Sequence transformer with skip connection
            seq_tfmr_in = torch.cat([
                node, self.trunk[f'skip_embed_{b}'](init_node)
            ], dim=-1)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                seq_tfmr_in.transpose(0, 1),
                src_key_padding_mask=(1 - mask).bool() if mask is not None else None
            ).transpose(0, 1)
            node = node + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            if mask is not None:
                node = node * mask[..., None]
        
        # Unscale rigids before computing scores
        curr_rigids = self._apply_trans_fn(curr_rigids, lambda x: x / self.coordinate_scaling)
        
        # Predict torsion angles from final node features
        unorm_angles, angles = self.angle_resnet(node, init_node)
        
        # Compute scores from rigids difference (SE3 Diffusion / DyneTrion approach)
        # This is the principled way to compute SE(3) scores
        pred_rot_score, pred_trans_score = self._compute_scores_from_rigids(
            curr_rigids, init_rigids, t
        )
        
        if mask is not None:
            pred_rot_score = pred_rot_score * mask[..., None]
            pred_trans_score = pred_trans_score * mask[..., None]
        
        return {
            'rigids': curr_rigids.to_tensor_7(),
            'init_rigids': init_rigids.to_tensor_7(),
            'node_repr': node,
            'edge_repr': edge,
            'pred_rot_score': pred_rot_score,
            'pred_trans_score': pred_trans_score,
            'angles': angles,
            'unorm_angles': unorm_angles,
        }
    
    @staticmethod
    def init_random_rigids(
        batch_size: int,
        num_res: int,
        trans_scale: float = 10.0,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Initialize random rigids for score-based generation."""
        if device is None:
            device = torch.device('cpu')
        
        # Random rotation: sample random quaternion
        quats = torch.randn(batch_size, num_res, 4, device=device)
        quats = quats / quats.norm(dim=-1, keepdim=True)
        
        # Random translation: centered gaussian with specified scale
        trans = torch.randn(batch_size, num_res, 3, device=device) * trans_scale
        
        rigids = torch.cat([quats, trans], dim=-1)
        return rigids
    
    def sample_structure(
        self,
        node_repr: torch.Tensor,
        edge_repr: torch.Tensor,
        num_steps: int = 100,
        mask: Optional[torch.Tensor] = None,
        trans_scale: float = 10.0,
    ) -> torch.Tensor:
        """Sample structure using Langevin dynamics / diffusion sampling.
        
        Args:
            node_repr: [B, N, C_s_input] sequence/node embeddings
            edge_repr: [B, N, N, C_z_input] pair/edge embeddings
            num_steps: number of denoising steps
            mask: [B, N] optional mask
            trans_scale: initial translation scale for random initialization
            
        Returns:
            [B, N, 7] sampled rigids
        """
        B, N = node_repr.shape[:2]
        device = node_repr.device
        
        # Initialize random rigids
        rigids = self.init_random_rigids(B, N, trans_scale, device)
        
        # Annealed Langevin dynamics
        for i in range(num_steps):
            t = torch.ones(B, device=device) * (1 - i / num_steps)
            
            with torch.no_grad():
                out = self.forward(node_repr, edge_repr, rigids, t, mask)
            
            # Update rigids using predicted scores
            rot_score = out['pred_rot_score']
            trans_score = out['pred_trans_score']
            
            # Step size (annealed)
            step_size = 0.01 * (1 - i / num_steps)
            
            # Update translation
            rigids[..., 4:] = rigids[..., 4:] + step_size * trans_score
            
            # Update rotation using rotation vectors
            rot_update = step_size * rot_score
            rot_quat = rotvec_to_quat(rot_update)
            curr_quat = rigids[..., :4]
            new_quat = quat_multiply(rot_quat, curr_quat)
            new_quat = new_quat / new_quat.norm(dim=-1, keepdim=True)
            rigids[..., :4] = new_quat
        
        return rigids


# Backwards compatibility alias
MinimalIPA = IPABlock
