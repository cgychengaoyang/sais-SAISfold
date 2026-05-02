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
PairEngram - Hash-based n-gram memory for pair representations.

This module implements DeepSeek-style O(1) hash-based n-gram memory
for augmenting pair representations in the Pairformer stack.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairEngram(nn.Module):
    """
    Hash-based n-gram memory for pair representations.
    
    Implements DeepSeek-style O(1) lookup with:
    - Separate embedding table for each (n-gram order, hash head) pair
    - Prime-based table sizes for better distribution
    - Multiplicative-XOR hash function
    - Position-aware hashing to reduce collisions
    
    Args:
        table_size: Number of slots per embedding table (default: 5000)
        slot_dim: Dimension of each slot (matches c_z = 128 for Protenix)
        ngram_orders: Tuple of n-gram orders to use (default: (3, 4, 5))
        num_hash_heads: Number of hash heads per n-gram order (default: 2)
        use_position_aware: Whether to use position-aware hashing (default: True)
    """
    
    @staticmethod
    def _next_prime(n: int) -> int:
        """Find next prime number >= n."""
        def is_prime(x):
            if x < 2:
                return False
            for i in range(2, int(x**0.5) + 1):
                if x % i == 0:
                    return False
            return True
        while not is_prime(n):
            n += 1
        return n
    
    def __init__(
        self,
        table_size: int = 5000,
        slot_dim: int = 128,
        ngram_orders: Tuple[int, ...] = (3, 4, 5),
        num_hash_heads: int = 2,
        use_position_aware: bool = True,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.ngram_orders = ngram_orders
        self.num_hash_heads = num_hash_heads
        self.use_position_aware = use_position_aware
        
        # Separate embedding table for each (n, k) pair with prime sizes
        self.engram_tables = nn.ParameterDict()
        self.table_sizes = {}
        
        for n in ngram_orders:
            for k in range(num_hash_heads):
                # Different prime size for each table for better distribution
                prime_size = self._next_prime(table_size + k * 997 + n * 101)
                table_key = f"E_{n}_{k}"
                self.engram_tables[table_key] = nn.Parameter(
                    torch.randn(prime_size, slot_dim) * 0.02
                )
                self.table_sizes[table_key] = prime_size
        
        self.total_slots = sum(self.table_sizes.values())
        
        # Hash multipliers for multiplicative-XOR hash
        golden_ratio = 0.618033988749895
        for n in ngram_orders:
            for k in range(num_hash_heads):
                multipliers = []
                for i in range(n):
                    val = int((2**31 - 1) * ((k + 1) * golden_ratio * (i + 1) % 1))
                    val = val | 1  # Make odd for better mixing
                    multipliers.append(val)
                
                self.register_buffer(
                    f"mult_{n}_{k}",
                    torch.tensor(multipliers, dtype=torch.int64)
                )
        
        # Position encoding primes for position-aware hashing
        if use_position_aware:
            for k in range(num_hash_heads):
                pos_prime = self._next_prime(997 + k * 100)
                self.register_buffer(
                    f"pos_prime_{k}",
                    torch.tensor(pos_prime, dtype=torch.int64)
                )
        
        # Context-aware gating network
        total_heads = len(ngram_orders) * num_hash_heads
        gate_input_dim = slot_dim * (total_heads + 1)
        self.gate_proj = nn.Sequential(
            nn.Linear(gate_input_dim, slot_dim),
            nn.LayerNorm(slot_dim),
            nn.Sigmoid(),
        )
        
        # Output projection
        self.out_proj = nn.Linear(slot_dim * total_heads, slot_dim)
        self.norm = nn.LayerNorm(slot_dim)
    
    def compute_hash(
        self,
        ngram_tokens: torch.Tensor,
        n: int,
        head: int,
        positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute DeepSeek-style multiplicative-XOR hash.
        
        Args:
            ngram_tokens: [B, N, N, n] token IDs for n-grams
            n: n-gram order
            head: hash head index
            positions: [B, N, N] position indices (optional)
            
        Returns:
            Hash indices [B, N, N]
        """
        multipliers = getattr(self, f"mult_{n}_{head}")  # [n]
        table_key = f"E_{n}_{head}"
        table_size = self.table_sizes[table_key]
        
        batch_size, n_tokens, _, n_tokens_n = ngram_tokens.shape
        device = ngram_tokens.device
        
        # Multiplicative-XOR hash
        mix = torch.zeros(batch_size, n_tokens, n_tokens, dtype=torch.int64, device=device)
        
        for i in range(n_tokens_n):
            token_i = ngram_tokens[:, :, :, i].long()
            mult_i = multipliers[i].item()
            mix = mix ^ (token_i * mult_i)
        
        # Position-aware hashing
        if self.use_position_aware and positions is not None:
            pos_prime = getattr(self, f"pos_prime_{head}")
            pos_hash = (positions.long() * pos_prime.item()) & 0x7FFFFFFF
            mix = mix ^ pos_hash
        
        hash_val = torch.abs(mix) % table_size
        return hash_val.long()
    
    def extract_ngrams(self, seq_tokens: torch.Tensor, n: int) -> torch.Tensor:
        """
        Extract n-grams from sequence tokens maintaining [B, N, N] shape for pairs.
        
        Args:
            seq_tokens: [B, N] sequence tokens
            n: n-gram order
            
        Returns:
            ngrams: [B, N, N, n] n-gram tokens for each pair
        """
        batch_size, seq_len = seq_tokens.shape
        
        # Pad sequence
        pad_left = (n - 1) // 2
        pad_right = (n - 1) - pad_left
        padded = F.pad(seq_tokens, (pad_left, pad_right), mode='constant', value=0)
        
        # Extract n-grams
        ngrams = padded.unfold(dimension=1, size=n, step=1)  # [B, N, n]
        
        # Expand for pair representation [B, N, N, n]
        # For pair (i, j), we use n-grams centered at both positions
        ngrams_i = ngrams.unsqueeze(2).expand(batch_size, seq_len, seq_len, n)
        ngrams_j = ngrams.unsqueeze(1).expand(batch_size, seq_len, seq_len, n)
        
        # Combine: concatenate n-grams from both positions
        return torch.cat([ngrams_i, ngrams_j], dim=-1)[:, :, :, :n]
    
    def forward(
        self,
        pair_repr: torch.Tensor,
        seq_tokens: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with hash-based n-gram lookup.
        
        Args:
            pair_repr: [B, N, N, c_z] pair representations
            seq_tokens: [B, N] sequence tokens
            pair_mask: [B, N, N] mask (optional)
            
        Returns:
            output: [B, N, N, c_z] enriched pair representations
            gate: [B, N, N, c_z] gating values
        """
        batch_size, n_tokens, _, dim = pair_repr.shape
        device = seq_tokens.device
        
        # Position indices for position-aware hashing
        if self.use_position_aware:
            positions_i = torch.arange(n_tokens, device=device).unsqueeze(0).unsqueeze(2)
            positions_j = torch.arange(n_tokens, device=device).unsqueeze(0).unsqueeze(1)
            positions = (positions_i + positions_j) % 1000  # Combine positions
            positions = positions.expand(batch_size, n_tokens, n_tokens)
        else:
            positions = None
        
        # Multi-head hash lookup
        all_embeddings = []
        
        for n in self.ngram_orders:
            ngrams = self.extract_ngrams(seq_tokens, n)  # [B, N, N, n]
            
            for h in range(self.num_hash_heads):
                # Compute hash
                hash_indices = self.compute_hash(ngrams, n, h, positions)  # [B, N, N]
                
                # Fetch from embedding table
                table_key = f"E_{n}_{h}"
                table = self.engram_tables[table_key]
                
                # Flatten and lookup
                flat_indices = hash_indices.view(-1)
                flat_indices = torch.clamp(flat_indices, 0, len(table) - 1)
                embeddings = table[flat_indices]  # [B*N*N, slot_dim]
                embeddings = embeddings.view(batch_size, n_tokens, n_tokens, self.slot_dim)
                
                all_embeddings.append(embeddings)
        
        # Concatenate all embeddings
        combined = torch.cat(all_embeddings, dim=-1)  # [B, N, N, H*slot_dim]
        
        # Output projection
        retrieved = self.out_proj(combined)  # [B, N, N, c_z]
        retrieved = self.norm(retrieved)
        
        # Context-aware gating
        gate_input = torch.cat([pair_repr] + all_embeddings, dim=-1)
        gate = self.gate_proj(gate_input)  # [B, N, N, c_z]
        
        # Gated combination
        output = gate * pair_repr + (1 - gate) * retrieved
        
        # Apply mask if provided
        if pair_mask is not None:
            output = output * pair_mask.unsqueeze(-1)
        
        return output, gate
