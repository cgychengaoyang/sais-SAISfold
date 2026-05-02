# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DeepSeek blocks for Pairformer replacement.

Implements DeepSeek-style attention blocks with optional Engram and MoE.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .pair_engram import PairEngram
from .pair_moe import PairMoE


class AdaptiveRMSNorm(nn.Module):
    """Adaptive RMSNorm with optional conditioning."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


class MultiHeadAttention(nn.Module):
    """Multi-head attention for pair representations."""
    
    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            q, k, v: [B, N, N, dim]
            mask: [B, N, N] (optional)
            
        Returns:
            output: [B, N, N, dim]
        """
        batch_size, n_tokens, _, dim = q.shape
        
        # Project and reshape
        q = self.q_proj(q).view(batch_size, n_tokens, n_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(k).view(batch_size, n_tokens, n_tokens, self.num_heads, self.head_dim)
        v = self.v_proj(v).view(batch_size, n_tokens, n_tokens, self.num_heads, self.head_dim)
        
        # Transpose for attention [B, H, N, N, d]
        q = q.permute(0, 3, 1, 2, 4)
        k = k.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 1, 2, 4)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N, N]
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(-1) == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply to values
        out = torch.matmul(attn, v)  # [B, H, N, N, d]
        
        # Reshape back
        out = out.permute(0, 2, 3, 1, 4).contiguous()
        out = out.view(batch_size, n_tokens, n_tokens, dim)
        
        return self.out_proj(out)


class DeepSeekPairBlock(nn.Module):
    """
    Single DeepSeek-style pair block with optional Engram and MoE.
    
    Args:
        dim: Dimension (c_z = 128)
        num_heads: Number of attention heads
        use_engram: Whether to use Engram in this block
        use_moe: Whether to use MoE in this block
        engram_config: Configuration for Engram (if use_engram)
        moe_config: Configuration for MoE (if use_moe)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        use_engram: bool = False,
        use_moe: bool = False,
        engram_config: Optional[dict] = None,
        moe_config: Optional[dict] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.use_engram = use_engram
        self.use_moe = use_moe
        
        # Pre-attention normalization
        self.norm1 = AdaptiveRMSNorm(dim)
        
        # Self-attention
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        
        # Optional Engram
        if use_engram:
            config = engram_config or {}
            self.engram = PairEngram(
                table_size=config.get('table_size', 5000),
                slot_dim=dim,
                ngram_orders=config.get('ngram_orders', (3, 4, 5)),
                num_hash_heads=config.get('num_hash_heads', 2),
            )
        
        # Post-attention normalization
        self.norm2 = AdaptiveRMSNorm(dim)
        
        # Optional MoE or standard FFN
        if use_moe:
            config = moe_config or {}
            self.moe = PairMoE(
                dim=dim,
                num_routed_experts=config.get('num_routed_experts', 64),
                num_shared_experts=config.get('num_shared_experts', 2),
                top_k=config.get('top_k', 2),
                expert_dim=config.get('expert_dim', 256),
                dropout=dropout,
            )
        else:
            # Standard FFN
            self.ffn = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
            )
    
    def forward(
        self,
        pair_repr: torch.Tensor,
        seq_tokens: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            pair_repr: [B, N, N, dim]
            seq_tokens: [B, N] sequence tokens for engram
            pair_mask: [B, N, N] mask
            
        Returns:
            output: [B, N, N, dim]
            aux_loss: Auxiliary loss from MoE (or None)
        """
        # Attention with residual
        normed = self.norm1(pair_repr)
        attn_out = self.attn(normed, normed, normed, pair_mask)
        pair_repr = pair_repr + attn_out
        
        # Optional Engram
        if self.use_engram:
            engram_out, _ = self.engram(pair_repr, seq_tokens, pair_mask)
            pair_repr = pair_repr + engram_out
        
        # MoE or FFN with residual
        normed = self.norm2(pair_repr)
        
        if self.use_moe:
            moe_out, aux_loss = self.moe(normed, pair_mask)
            pair_repr = pair_repr + moe_out
        else:
            pair_repr = pair_repr + self.ffn(normed)
            aux_loss = None
        
        return pair_repr, aux_loss


class DeepSeekPairStack(nn.Module):
    """
    Stack of DeepSeek pair blocks with shared Engram.
    
    Args:
        n_blocks: Number of blocks
        dim: Dimension (c_z = 128)
        num_heads: Number of attention heads
        engram_layers: List of layer indices to add Engram (e.g., [0, 2, 4, 6])
        moe_layers: List of layer indices to add MoE
        engram_config: Configuration for Engram
        moe_config: Configuration for MoE
        dropout: Dropout rate
        use_gradient_checkpointing: Whether to use gradient checkpointing
    """
    
    def __init__(
        self,
        n_blocks: int = 8,
        dim: int = 128,
        num_heads: int = 4,
        engram_layers: Optional[list] = None,
        moe_layers: Optional[list] = None,
        engram_config: Optional[dict] = None,
        moe_config: Optional[dict] = None,
        dropout: float = 0.0,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.dim = dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Default: every other layer gets engram and moe
        if engram_layers is None:
            engram_layers = list(range(0, n_blocks, 2))
        if moe_layers is None:
            moe_layers = list(range(0, n_blocks, 2))
        
        self.engram_layers = set(engram_layers)
        self.moe_layers = set(moe_layers)
        
        # Shared Engram for all layers that use it
        self.shared_engram = None
        if len(self.engram_layers) > 0:
            config = engram_config or {}
            self.shared_engram = PairEngram(
                table_size=config.get('table_size', 5000),
                slot_dim=dim,
                ngram_orders=config.get('ngram_orders', (3, 4, 5)),
                num_hash_heads=config.get('num_hash_heads', 2),
            )
        
        # Build blocks
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            use_engram = i in self.engram_layers
            use_moe = i in self.moe_layers
            
            self.blocks.append(
                DeepSeekPairBlock(
                    dim=dim,
                    num_heads=num_heads,
                    use_engram=False,  # We use shared engram manually
                    use_moe=use_moe,
                    moe_config=moe_config,
                    dropout=dropout,
                )
            )
    
    def forward(
        self,
        pair_repr: torch.Tensor,
        seq_tokens: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all blocks.
        
        Args:
            pair_repr: [B, N, N, dim]
            seq_tokens: [B, N] sequence tokens
            pair_mask: [B, N, N] mask
            
        Returns:
            output: [B, N, N, dim]
            total_aux_loss: Sum of all MoE auxiliary losses
        """
        total_aux_loss = torch.tensor(0.0, device=pair_repr.device)
        
        for i, block in enumerate(self.blocks):
            # Use shared engram before this block if configured
            if i in self.engram_layers and self.shared_engram is not None:
                engram_out, _ = self.shared_engram(pair_repr, seq_tokens, pair_mask)
                pair_repr = pair_repr + engram_out
            
            # Block forward
            if self.use_gradient_checkpointing and self.training and pair_repr.requires_grad:
                pair_repr, aux_loss = checkpoint(
                    block, pair_repr, seq_tokens, pair_mask,
                    use_reentrant=False,
                )
            else:
                pair_repr, aux_loss = block(pair_repr, seq_tokens, pair_mask)
            
            # Accumulate auxiliary loss
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss
        
        return pair_repr, total_aux_loss


class DeepSeekPairStackWithSingles(nn.Module):
    """
    DeepSeek PairStack with singles update, replacing PairformerStack.
    
    This is the main module that can be used as a drop-in replacement
    for Protenix's PairformerStack.
    
    Args:
        c_s: Single representation dimension (384)
        c_z: Pair representation dimension (128)
        n_blocks: Number of blocks (default: 8)
        num_heads: Number of attention heads (default: 4)
        engram_layers: Layers to use Engram (default: [0, 2, 4, 6])
        moe_layers: Layers to use MoE (default: [0, 2, 4, 6])
        engram_config: Engram configuration
        moe_config: MoE configuration
        dropout: Dropout rate
        use_gradient_checkpointing: Whether to use gradient checkpointing
    """
    
    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        n_blocks: int = 8,
        num_heads: int = 4,
        engram_layers: Optional[list] = None,
        moe_layers: Optional[list] = None,
        engram_config: Optional[dict] = None,
        moe_config: Optional[dict] = None,
        dropout: float = 0.0,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        
        # Pair stack
        self.pair_stack = DeepSeekPairStack(
            n_blocks=n_blocks,
            dim=c_z,
            num_heads=num_heads,
            engram_layers=engram_layers,
            moe_layers=moe_layers,
            engram_config=engram_config,
            moe_config=moe_config,
            dropout=dropout,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        
        # Singles update from pairs
        from .singles_update import PairToSingleAttention, SingleTransition
        
        self.singles_attn = nn.ModuleList([
            PairToSingleAttention(c_s, c_z, num_heads, dropout)
            for _ in range(n_blocks)
        ])
        
        self.singles_transition = nn.ModuleList([
            SingleTransition(c_s, dropout)
            for _ in range(n_blocks)
        ])
    
    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        seq_tokens: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
        single_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass, compatible with PairformerStack interface.
        
        Args:
            s: [B, N, c_s] single representations
            z: [B, N, N, c_z] pair representations
            seq_tokens: [B, N] sequence tokens
            pair_mask: [B, N, N] pair mask
            single_mask: [B, N] single mask
            
        Returns:
            s: [B, N, c_s] updated single representations
            z: [B, N, N, c_z] updated pair representations
            aux_loss: MoE auxiliary loss
        """
        # Process through pair stack
        z, aux_loss = self.pair_stack(z, seq_tokens, pair_mask)
        
        # Update singles from pairs for each block
        for attn, transition in zip(self.singles_attn, self.singles_transition):
            s = attn(s, z, single_mask)
            s = transition(s, single_mask)
        
        return s, z, aux_loss
