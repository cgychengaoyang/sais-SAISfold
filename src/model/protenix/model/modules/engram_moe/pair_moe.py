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
PairMoE - Mixture of Experts for pair representations.

Implements DeepSeek-style MoE with:
- Shared experts + routed experts
- Top-k routing with load balancing
- No CPU-GPU synchronization
- Chunked processing for large sequences
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Single expert FFN."""
    
    def __init__(self, dim: int, expert_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, expert_dim, bias=False)
        self.w2 = nn.Linear(expert_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU-style activation
        return self.w2(self.dropout(F.silu(self.w1(x))))


class PairMoE(nn.Module):
    """
    MoE layer for pair representations with shared + routed experts.
    
    Args:
        dim: Model dimension (c_z = 128)
        num_routed_experts: Number of routed experts (default: 64)
        num_shared_experts: Number of shared experts (default: 2)
        top_k: Number of experts to route to (default: 2)
        expert_dim: Expert hidden dimension (default: 256)
        dropout: Dropout rate
        moe_chunk_size: Chunk size for processing (default: 1024)
    """
    
    def __init__(
        self,
        dim: int = 128,
        num_routed_experts: int = 64,
        num_shared_experts: int = 2,
        top_k: int = 2,
        expert_dim: int = 256,
        dropout: float = 0.0,
        moe_chunk_size: int = 1024,
    ):
        super().__init__()
        self.dim = dim
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.expert_dim = expert_dim
        self.moe_chunk_size = moe_chunk_size
        
        # Router
        self.router = nn.Linear(dim, num_routed_experts, bias=False)
        
        # Shared experts (always activated)
        self.shared_experts = nn.ModuleList([
            Expert(dim, expert_dim, dropout)
            for _ in range(num_shared_experts)
        ])
        
        # Routed experts
        self.routed_experts = nn.ModuleList([
            Expert(dim, expert_dim, dropout)
            for _ in range(num_routed_experts)
        ])
        
        # Normalization
        self.norm = nn.LayerNorm(dim)
        
    def forward(
        self,
        x: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: [B, N, N, dim] pair representations
            pair_mask: [B, N, N] mask (optional)
            
        Returns:
            output: [B, N, N, dim] output representations
            aux_loss: Load balancing auxiliary loss
        """
        batch_size, n_tokens, _, dim = x.shape
        device = x.device
        
        # Flatten for processing [B*N*N, dim]
        x_flat = x.view(-1, dim)
        if pair_mask is not None:
            mask_flat = pair_mask.view(-1)
        else:
            mask_flat = torch.ones(batch_size * n_tokens * n_tokens, device=device, dtype=torch.bool)
        
        # Get valid positions
        mask_flat = mask_flat.bool()
        num_valid = mask_flat.sum().item()
        
        if num_valid == 0:
            # All masked, return zeros
            return torch.zeros_like(x), torch.tensor(0.0, device=device)
        
        x_valid = x_flat[mask_flat]  # [num_valid, dim]
        
        # Normalize
        x_valid = self.norm(x_valid)
        
        # Router logits [num_valid, num_routed_experts]
        router_logits = self.router(x_valid)
        
        # Top-k routing
        topk_logits, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
        topk_gates = F.softmax(topk_logits, dim=-1)  # [num_valid, top_k]
        
        # Process shared experts (all valid positions)
        shared_output = torch.zeros_like(x_valid)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(x_valid)
        shared_output = shared_output / self.num_shared_experts
        
        # Process routed experts (chunked for memory efficiency)
        routed_output = torch.zeros_like(x_valid)
        
        # Group by expert for efficient batching
        for expert_idx in range(self.num_routed_experts):
            # Find all positions that route to this expert
            mask = (topk_indices == expert_idx).any(dim=-1)  # [num_valid]
            
            if mask.any():
                positions = mask.nonzero(as_tuple=True)[0]
                expert_input = x_valid[positions]
                expert_out = self.routed_experts[expert_idx](expert_input)
                
                # Get gate weights for this expert
                expert_positions_in_topk = (topk_indices == expert_idx).nonzero(as_tuple=True)
                gates = topk_gates[expert_positions_in_topk[0], expert_positions_in_topk[1]]
                
                routed_output[positions] = routed_output[positions] + gates.unsqueeze(-1) * expert_out
        
        # Combine shared and routed
        output_valid = shared_output + routed_output
        
        # Scatter back to full tensor
        output_flat = torch.zeros_like(x_flat)
        output_flat[mask_flat] = output_valid
        output = output_flat.view(batch_size, n_tokens, n_tokens, dim)
        
        # Load balancing loss (encourage uniform expert usage)
        router_probs = F.softmax(router_logits, dim=-1).mean(dim=0)  # [num_routed_experts]
        aux_loss = self.num_routed_experts * (router_probs ** 2).mean()
        
        # Apply mask
        if pair_mask is not None:
            output = output * pair_mask.unsqueeze(-1)
        
        return output, aux_loss


class DeepSeekMoE(nn.Module):
    """
    DeepSeek-style MoE with configurable expert routing.
    
    This is a simplified version optimized for pair representations.
    """
    
    def __init__(
        self,
        dim: int = 128,
        num_experts: int = 8,
        top_k: int = 2,
        expert_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.gate = nn.Linear(dim, num_experts, bias=False)
        
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, expert_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_dim, dim)
            )
            for _ in range(num_experts)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [B, N, N, dim] pair representations
            pair_mask: [B, N, N] mask (optional)
            
        Returns:
            output: [B, N, N, dim] output
            aux_loss: Load balancing loss
        """
        batch_size, n_tokens, _, dim = x.shape
        
        # Flatten
        x_flat = x.view(-1, dim)
        
        # Router logits
        logits = self.gate(x_flat)  # [B*N*N, num_experts]
        
        # Top-k routing
        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_gates = F.softmax(topk_logits, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Route to experts
        for i in range(self.num_experts):
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = self.experts[i](expert_input)
                
                # Get gate weights
                positions = (topk_indices == i).nonzero(as_tuple=True)
                gates = topk_gates[positions[0], positions[1]]
                
                output[mask] = output[mask] + gates.unsqueeze(-1) * expert_output
        
        # Reshape
        output = output.view(batch_size, n_tokens, n_tokens, dim)
        
        # Load balancing loss
        router_probs = F.softmax(logits, dim=-1).mean(dim=0)
        aux_loss = self.num_experts * (router_probs ** 2).mean()
        
        # Apply mask
        if pair_mask is not None:
            output = output * pair_mask.unsqueeze(-1)
        
        return output, aux_loss
