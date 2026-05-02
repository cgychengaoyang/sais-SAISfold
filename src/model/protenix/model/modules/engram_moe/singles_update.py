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
Singles update modules for integrating pair information into single representations.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairToSingleAttention(nn.Module):
    """
    Attention from singles to pairs to update single representations.
    
    Similar to AlphaFold's approach of using pair biases in attention.
    """
    
    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        
        # Single projections
        self.q_proj = nn.Linear(c_s, c_s, bias=False)
        self.kv_proj = nn.Linear(c_s, c_s * 2, bias=False)
        self.out_proj = nn.Linear(c_s, c_s, bias=False)
        
        # Pair bias projection
        self.pair_bias_proj = nn.Linear(c_z, num_heads, bias=False)
        
        # Output projection for aggregated pairs
        self.pair_agg_proj = nn.Linear(c_z, c_s, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Update singles using pair information.
        
        Args:
            s: [B, N, c_s] single representations
            z: [B, N, N, c_z] pair representations
            mask: [B, N] mask for singles
            
        Returns:
            updated_s: [B, N, c_s]
        """
        batch_size, n_tokens, _ = s.shape
        
        # Project singles
        q = self.q_proj(s)  # [B, N, c_s]
        kv = self.kv_proj(s)  # [B, N, c_s*2]
        k, v = kv.chunk(2, dim=-1)  # [B, N, c_s] each
        
        # Reshape for multi-head attention
        q = q.view(batch_size, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, H, N, d]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        
        # Add pair bias
        pair_bias = self.pair_bias_proj(z)  # [B, N, N, H]
        pair_bias = pair_bias.permute(0, 3, 1, 2)  # [B, H, N, N]
        scores = scores + pair_bias
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(-1) == 0, float('-inf'))
        
        # Attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, H, N, d]
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, n_tokens, self.c_s)
        out = self.out_proj(out)
        
        # Aggregate pair information
        # Average over the second dimension (j) to get per-i pair features
        pair_agg = z.mean(dim=2)  # [B, N, c_z]
        pair_agg = self.pair_agg_proj(pair_agg)  # [B, N, c_s]
        
        # Combine with residual
        s = s + out + pair_agg
        
        return s


class SingleTransition(nn.Module):
    """
    Transition layer for single representations.
    
    Simple FFN with GELU activation.
    """
    
    def __init__(
        self,
        c_s: int = 384,
        dropout: float = 0.0,
        expansion_factor: int = 4,
    ):
        super().__init__()
        self.c_s = c_s
        
        self.norm = nn.LayerNorm(c_s)
        self.ffn = nn.Sequential(
            nn.Linear(c_s, c_s * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_s * expansion_factor, c_s),
        )
        
    def forward(
        self,
        s: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            s: [B, N, c_s]
            mask: [B, N] (unused, for compatibility)
            
        Returns:
            updated_s: [B, N, c_s]
        """
        s = s + self.ffn(self.norm(s))
        return s
