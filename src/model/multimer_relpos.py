import torch
import torch.nn as nn
import torch.nn.functional as F


class ChainRelativePositionalEncoding(nn.Module):
    """
    AlphaFold-Multimer / OpenFold style chain-relative positional encoding.

    Inputs:
        residue_index: [B, N] (per-chain residue counter, e.g. 1..L_chain)
        asym_id:       [B, N] chain id
        entity_id:     [B, N] sequence identity id (homomers share the same entity_id)
        sym_id:        [B, N] symmetry copy id within entity
    Output:
        [B, N, N, c_z] pair bias
    """
    def __init__(self, c_z: int, max_relative_idx: int = 32, max_relative_chain: int = 2):
        super().__init__()
        self.max_relative_idx = int(max_relative_idx)
        self.max_relative_chain = int(max_relative_chain)

        relpos_bins = 2 * self.max_relative_idx + 2           # +1 for out-of-range, +1 for inter-chain bin
        relchain_bins = 2 * self.max_relative_chain + 2       # +1 for out-of-range, +1 for different-entity bin
        self.no_bins = relpos_bins + 1 + relchain_bins        # +1 for entity_same flag

        self.linear = nn.Linear(self.no_bins, c_z, bias=False)

    def forward(
        self,
        residue_index: torch.Tensor,  # [B, N]
        asym_id: torch.Tensor,        # [B, N]
        entity_id: torch.Tensor,      # [B, N]
        sym_id: torch.Tensor,         # [B, N]
    ) -> torch.Tensor:
        B, N = residue_index.shape

        # same chain?
        same_chain = (asym_id[:, :, None] == asym_id[:, None, :])  # [B, N, N]

        # residue offset (only meaningful within a chain)
        offset = residue_index[:, :, None] - residue_index[:, None, :]  # [B, N, N]
        offset = torch.clamp(offset + self.max_relative_idx, 0, 2 * self.max_relative_idx)
        inter_chain_bin = torch.full_like(offset, 2 * self.max_relative_idx + 1)
        final_offset = torch.where(same_chain, offset, inter_chain_bin)  # [B,N,N]
        relpos_oh = F.one_hot(final_offset.long(), num_classes=2 * self.max_relative_idx + 2).float()

        # same entity?
        same_entity = (entity_id[:, :, None] == entity_id[:, None, :])  # [B,N,N]
        entity_feat = same_entity[..., None].float()

        # sym_id relative offset (only meaningful within an entity)
        chain_offset = sym_id[:, :, None] - sym_id[:, None, :]  # [B,N,N]
        chain_offset = torch.clamp(chain_offset + self.max_relative_chain, 0, 2 * self.max_relative_chain)
        diff_entity_bin = torch.full_like(chain_offset, 2 * self.max_relative_chain + 1)
        final_chain_offset = torch.where(same_entity, chain_offset, diff_entity_bin)
        relchain_oh = F.one_hot(final_chain_offset.long(), num_classes=2 * self.max_relative_chain + 2).float()

        feat = torch.cat([relpos_oh, entity_feat, relchain_oh], dim=-1)  # [B,N,N,no_bins]
        return self.linear(feat)  # [B,N,N,c_z]


class ChainIdEmbedding(nn.Module):
    """
    Simple single-feature embedding from (asym_id, entity_id, sym_id).
    Output shape: [B, N, c_s]
    """
    def __init__(
        self,
        c_s: int,
        max_asym_id: int = 64,
        max_entity_id: int = 64,
        max_sym_id: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_asym_id = int(max_asym_id)
        self.max_entity_id = int(max_entity_id)
        self.max_sym_id = int(max_sym_id)
        self.asym = nn.Embedding(self.max_asym_id + 1, c_s)
        self.entity = nn.Embedding(self.max_entity_id + 1, c_s)
        self.sym = nn.Embedding(self.max_sym_id + 1, c_s)
        self.dropout = nn.Dropout(dropout)

    def forward(self, asym_id: torch.Tensor, entity_id: torch.Tensor, sym_id: torch.Tensor) -> torch.Tensor:
        asym_id = asym_id.clamp(min=0, max=self.max_asym_id)
        entity_id = entity_id.clamp(min=0, max=self.max_entity_id)
        sym_id = sym_id.clamp(min=0, max=self.max_sym_id)
        x = self.asym(asym_id) + self.entity(entity_id) + self.sym(sym_id)
        return self.dropout(x)