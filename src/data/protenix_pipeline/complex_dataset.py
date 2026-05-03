"""Multi-chain complex dataset and collate for batched Protenix training."""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
import numpy as np


class ProtenixComplexDataset(Dataset):
    """Dataset that loads preprocessed Protenix complex .pt files.
    
    Each sample is a dict produced by preprocess_pdb_protenix_full() for an
    entire multi-chain structure.
    """

    def __init__(self, pt_paths: List[str]):
        self.pt_paths = pt_paths

    def __len__(self) -> int:
        return len(self.pt_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = torch.load(self.pt_paths[idx], map_location="cpu")
        return data


def _pad_tensor(t: torch.Tensor, target: int, dim: int, pad_val) -> torch.Tensor:
    if t.shape[dim] == target:
        return t
    pad_size = target - t.shape[dim]
    pad_shape = list(t.shape)
    pad_shape[dim] = pad_size
    pad = torch.full(pad_shape, pad_val, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=dim)


def _get_pad_value(t: torch.Tensor) -> Any:
    if t.dtype == torch.bool:
        return False
    if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return 0
    return 0.0


def collate_protenix_complex(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of complex samples into a batched dict.
    
    Pads token-level, atom-level, token-pair-level, atom-pair-level, and
    block-level features to the maximum sizes across the batch.
    """
    B = len(batch)
    max_tokens = max(b["aatype"].shape[0] for b in batch)
    max_atoms = max(b["input_feature_dict"]["ref_pos"].shape[0] for b in batch)
    max_blocks = max(b["input_feature_dict"]["d_lm"].shape[0] for b in batch)

    # ------------------------------------------------------------------
    # Helper closures for padding
    # ------------------------------------------------------------------
    def pad_token(t: torch.Tensor, pad_val=None):
        if pad_val is None:
            pad_val = _get_pad_value(t)
        return _pad_tensor(t, max_tokens, 0, pad_val)

    def pad_token_dim1(t: torch.Tensor, pad_val=None):
        """For tensors like [1, N] -> pad dim 1 to max_tokens."""
        if pad_val is None:
            pad_val = _get_pad_value(t)
        return _pad_tensor(t, max_tokens, 1, pad_val)

    def pad_token_pair(t: torch.Tensor, pad_val=None):
        if pad_val is None:
            pad_val = _get_pad_value(t)
        t = _pad_tensor(t, max_tokens, 0, pad_val)
        t = _pad_tensor(t, max_tokens, 1, pad_val)
        return t

    def pad_atom(t: torch.Tensor, pad_val=None):
        if pad_val is None:
            pad_val = _get_pad_value(t)
        return _pad_tensor(t, max_atoms, 0, pad_val)

    def pad_atom_pair(t: torch.Tensor, pad_val=None):
        if pad_val is None:
            pad_val = _get_pad_value(t)
        t = _pad_tensor(t, max_atoms, 0, pad_val)
        t = _pad_tensor(t, max_atoms, 1, pad_val)
        return t

    def pad_block(t: torch.Tensor, pad_val=None):
        if pad_val is None:
            pad_val = _get_pad_value(t)
        return _pad_tensor(t, max_blocks, 0, pad_val)

    # ------------------------------------------------------------------
    # Collate input_feature_dict keys
    # ------------------------------------------------------------------
    input_feature_keys = set(batch[0]["input_feature_dict"].keys())
    for b in batch[1:]:
        input_feature_keys |= set(b["input_feature_dict"].keys())

    # Build batched input_feature_dict
    batched_ifd: Dict[str, Any] = {}
    masks: List[torch.Tensor] = []

    for key in input_feature_keys:
        samples = [b["input_feature_dict"].get(key) for b in batch]
        # Skip if any sample is missing this key
        if any(s is None for s in samples):
            continue
        first = samples[0]

        # Determine padding strategy based on shape
        if isinstance(first, dict):
            # Nested dict, e.g. constraint_feature or pad_info
            if key == "constraint_feature":
                batched_ifd[key] = {
                    subkey: torch.stack([pad_token_pair(s[subkey]) for s in samples])
                    for subkey in first.keys()
                }
            elif key == "pad_info":
                batched_pad_info: Dict[str, Any] = {}
                for subkey in first.keys():
                    if subkey == "mask_trunked":
                        batched_pad_info[subkey] = torch.stack(
                            [pad_block(s[subkey]) for s in samples]
                        )
                    else:
                        # scalar ints per sample
                        batched_pad_info[subkey] = torch.tensor(
                            [int(s[subkey]) for s in samples], dtype=torch.int64
                        )
                batched_ifd[key] = batched_pad_info
            else:
                raise ValueError(f"Unsupported nested dict key: {key}")
        elif isinstance(first, torch.Tensor):
            shape = first.shape
            # Strategy dispatch
            if key in ("restype", "profile", "deletion_mean", "has_frame",
                       "frame_atom_index", "token_index", "residue_index",
                       "asym_id", "entity_id", "sym_id"):
                # Token-level, pad dim 0
                batched_ifd[key] = torch.stack([pad_token(s) for s in samples])
            elif key in ("ref_pos", "ref_charge", "ref_mask", "ref_element",
                         "ref_atom_name_chars", "ref_space_uid", "atom_to_token_idx",
                         "atom_to_tokatom_idx", "is_protein", "is_ligand",
                         "is_dna", "is_rna", "mol_id", "mol_atom_index",
                         "entity_mol_id", "pae_rep_atom_mask", "plddt_m_rep_atom_mask",
                         "distogram_rep_atom_mask", "modified_res_mask"):
                # Atom-level, pad dim 0
                batched_ifd[key] = torch.stack([pad_atom(s) for s in samples])
            elif key in ("token_bonds", "relp"):
                # Token-pair-level, pad dim 0 and 1
                batched_ifd[key] = torch.stack([pad_token_pair(s) for s in samples])
            elif key == "bond_mask":
                # Atom-pair-level, pad dim 0 and 1
                batched_ifd[key] = torch.stack([pad_atom_pair(s) for s in samples])
            elif key in ("msa", "has_deletion", "deletion_value"):
                # [1, N] or similar -> pad dim 1
                batched_ifd[key] = torch.stack([pad_token_dim1(s) for s in samples])
            elif key in ("template_restype",):
                # [4, N] -> pad dim 1
                batched_ifd[key] = torch.stack([pad_token_dim1(s) for s in samples])
            elif key in ("template_all_atom_mask",):
                # [4, N, 37] -> pad dim 1
                batched_ifd[key] = torch.stack([pad_token_dim1(s) for s in samples])
            elif key in ("template_all_atom_positions",):
                # [4, N, 37, 3] -> pad dim 1
                batched_ifd[key] = torch.stack([pad_token_dim1(s) for s in samples])
            elif key in ("d_lm",):
                # [n_blocks, 32, 128, 3] -> pad dim 0
                batched_ifd[key] = torch.stack([pad_block(s) for s in samples])
            elif key in ("v_lm",):
                # [n_blocks, 32, 128, 1] -> pad dim 0
                batched_ifd[key] = torch.stack([pad_block(s) for s in samples])
            elif key == "resolution":
                # [1] or scalar -> squeeze and stack
                batched_ifd[key] = torch.stack([s.reshape(()) for s in samples])
            elif len(shape) == 0:
                # Scalar tensor -> just stack
                batched_ifd[key] = torch.stack([s for s in samples])
            else:
                # Fallback: inspect first dim
                if shape[0] == first.shape[0]:
                    # Try to infer from dimension
                    n_tokens = batch[0]["aatype"].shape[0]
                    n_atoms = batch[0]["input_feature_dict"]["ref_pos"].shape[0]
                    if shape[0] == n_tokens:
                        batched_ifd[key] = torch.stack([pad_token(s) for s in samples])
                    elif shape[0] == n_atoms:
                        batched_ifd[key] = torch.stack([pad_atom(s) for s in samples])
                    else:
                        raise ValueError(f"Cannot infer padding for key {key} with shape {shape}")
                else:
                    raise ValueError(f"Cannot infer padding for key {key} with shape {shape}")
        else:
            raise ValueError(f"Unsupported type for key {key}: {type(first)}")

    # ------------------------------------------------------------------
    # Build padding mask based on original token counts
    # ------------------------------------------------------------------
    token_counts = [b["aatype"].shape[0] for b in batch]
    mask = torch.zeros(B, max_tokens, dtype=torch.float32)
    for i, n in enumerate(token_counts):
        mask[i, :n] = 1.0

    # ------------------------------------------------------------------
    # Collate training labels
    # ------------------------------------------------------------------
    def stack_label(key: str, pad_fn):
        return torch.stack([pad_fn(b[key]) for b in batch])

    batched_aatype = stack_label("aatype", pad_token)

    # rigids_0 is a [N, 7] tensor: [quat(4), trans(3)]
    def pad_rigids(rigids: torch.Tensor) -> torch.Tensor:
        n = rigids.shape[0]
        if n == max_tokens:
            return rigids
        pad = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=rigids.dtype,
            device=rigids.device,
        ).expand(max_tokens - n, -1)
        return torch.cat([rigids, pad], dim=0)

    batched_rigids = torch.stack([pad_rigids(b["rigids_0"]) for b in batch])

    batched_torsion = stack_label("torsion_angles_sin_cos", pad_token)
    batched_alt_torsion = stack_label("alt_torsion_angles_sin_cos", pad_token)
    batched_torsion_mask = stack_label("torsion_angles_mask", pad_token)

    # Optional keys
    batched_atom37_pos = None
    batched_atom37_mask = None
    if "atom37_pos" in batch[0]:
        batched_atom37_pos = torch.stack([pad_token(b["atom37_pos"]) for b in batch])
    if "atom37_mask" in batch[0]:
        batched_atom37_mask = torch.stack([pad_token(b["atom37_mask"]) for b in batch])

    result = {
        "input_feature_dict": batched_ifd,
        "aatype": batched_aatype,
        "rigids_0": batched_rigids,
        "torsion_angles_sin_cos": batched_torsion,
        "alt_torsion_angles_sin_cos": batched_alt_torsion,
        "torsion_angles_mask": batched_torsion_mask,
        "mask": mask,
    }
    if batched_atom37_pos is not None:
        result["atom37_pos"] = batched_atom37_pos
    if batched_atom37_mask is not None:
        result["atom37_mask"] = batched_atom37_mask

    return result
