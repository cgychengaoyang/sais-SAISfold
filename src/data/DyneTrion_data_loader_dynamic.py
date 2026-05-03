"""PDB dataset loader."""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import math
from typing import Optional
from src.experiments import utils as eu
import torch
import torch.distributed as dist

import tree
import numpy as np
import torch
import pandas as pd
import logging
import random
import functools as fn
from src.data import se3_diffuser
from torch.utils import data
from src.data import utils as du
from openfold.data import data_transforms
from openfold.np import residue_constants
from openfold.utils import rigid_utils
from src.data.residue_constants import is_rna_residue
# from data import pdb_data_loader
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def _np_load_allow_pickle_compat(path: str):
    """
    Workaround for loading .npz that contains pickled python objects created by
    a different NumPy version (e.g., numpy 2.x pickles refer to 'numpy._core').
    This registers module aliases so pickle can import them successfully.
    """
    try:
        import numpy.core as _npcore
        import numpy.core.multiarray as _ncmulti
        import numpy.core._multiarray_umath as _nmu

        # Alias for numpy 2.x pickle module paths
        sys.modules.setdefault("numpy._core", _npcore)
        sys.modules.setdefault("numpy._core.multiarray", _ncmulti)
        sys.modules.setdefault("numpy._core._multiarray_umath", _nmu)
    except Exception:
        # If anything goes wrong, fall back to normal behavior
        pass

    return np.load(path, allow_pickle=True)


def _index_select_any(x, axis, idx):
    if torch.is_tensor(x):
        return torch.index_select(x, axis, idx.to(x.device))
    else:
        return np.take(x, idx.cpu().numpy(), axis=axis)


def _randint(lower, upper, generator, device):
    return int(
        torch.randint(
            lower,
            upper + 1,
            (1,),
            device=device,
            generator=generator,
        )[0]
    )


def _remap_asym_id_to_pdb_chain_index(asym_id_1d: torch.Tensor) -> torch.Tensor:
    """Remap arbitrary asym_id values to contiguous 0..K-1 chain indices."""
    asym_id_1d = torch.as_tensor(asym_id_1d, dtype=torch.long).view(-1)
    mapped = torch.empty_like(asym_id_1d)
    asym_to_chain = {}
    next_chain = 0
    for i, asym in enumerate(asym_id_1d.tolist()):
        if asym not in asym_to_chain:
            asym_to_chain[asym] = next_chain
            next_chain += 1
        mapped[i] = asym_to_chain[asym]
    return mapped


def _crop_feature_dict_by_residue_idx(feats: dict, crop_idx: torch.Tensor, original_n_res: int):
    """Crop features with fixed residue-axis convention for this project."""
    out = {}
    crop_idx = crop_idx.long()

    for k, v in feats.items():
        if not (torch.is_tensor(v) or isinstance(v, np.ndarray)):
            out[k] = v
            continue

        if k == "node_repr":
            if len(v.shape) < 1 or v.shape[0] != original_n_res:
                raise ValueError(
                    f"node_repr expects residue axis at dim 0 with size {original_n_res}, got shape {tuple(v.shape)}"
                )
            out[k] = _index_select_any(v, 0, crop_idx)
            continue

        if k == "edge_repr":
            if len(v.shape) < 2 or v.shape[0] != original_n_res or v.shape[1] != original_n_res:
                raise ValueError(
                    f"edge_repr expects residue axes at dims 0/1 with size {original_n_res}, got shape {tuple(v.shape)}"
                )
            tmp = _index_select_any(v, 0, crop_idx)
            out[k] = _index_select_any(tmp, 1, crop_idx)
            continue

        # All other residue-level features use dim 1 as residue axis.
        if len(v.shape) >= 2 and v.shape[1] == original_n_res:
            out[k] = _index_select_any(v, 1, crop_idx)
            continue

        out[k] = v

    return out


def _get_interface_residues(positions, atom_mask, asym_id, interface_threshold):
    coord_diff = positions[..., None, :, :] - positions[..., None, :, :, :]
    pairwise_dists = torch.sqrt(torch.sum(coord_diff ** 2, dim=-1))

    diff_chain_mask = (asym_id[..., None, :] != asym_id[..., :, None]).float()
    pair_mask = atom_mask[..., None, :] * atom_mask[..., None, :, :]
    mask = (diff_chain_mask[..., None] * pair_mask).bool()

    min_dist_per_res, _ = torch.where(mask, pairwise_dists, torch.inf).min(dim=-1)
    valid_interfaces = torch.sum((min_dist_per_res < interface_threshold).float(), dim=-1)
    interface_residue_idxs = torch.nonzero(valid_interfaces, as_tuple=True)[0]
    return interface_residue_idxs


def _get_contiguous_crop_idx(asym_id, seq_length, crop_size, generator):
    _, chain_idxs, chain_lens = asym_id.unique(
        dim=-1,
        return_inverse=True,
        return_counts=True,
    )

    shuffle_idx = torch.randperm(
        chain_lens.shape[-1],
        device=chain_lens.device,
        generator=generator,
    )

    _, idx_sorted = torch.sort(chain_idxs, stable=True)
    cum_sum = chain_lens.cumsum(dim=0)
    cum_sum = torch.cat(
        (torch.tensor([0], device=cum_sum.device, dtype=cum_sum.dtype), cum_sum[:-1]),
        dim=0,
    )
    asym_offsets = idx_sorted[cum_sum]

    num_budget = crop_size
    num_remaining = int(seq_length)

    crop_idxs = []
    for idx in shuffle_idx:
        chain_len = int(chain_lens[idx])
        num_remaining -= chain_len

        crop_size_max = min(num_budget, chain_len)
        crop_size_min = min(chain_len, max(0, num_budget - num_remaining))
        chain_crop_size = _randint(
            lower=crop_size_min,
            upper=crop_size_max,
            generator=generator,
            device=chain_lens.device,
        )
        num_budget -= chain_crop_size

        chain_start = _randint(
            lower=0,
            upper=chain_len - chain_crop_size,
            generator=generator,
            device=chain_lens.device,
        )

        asym_offset = asym_offsets[idx]
        crop_idxs.append(
            torch.arange(
                asym_offset + chain_start,
                asym_offset + chain_start + chain_crop_size,
                device=asym_offsets.device,
            )
        )

    return torch.concat(crop_idxs).sort().values


def _get_spatial_crop_idx(positions, atom_mask, asym_id, crop_size, interface_threshold, generator):
    interface_residues = _get_interface_residues(
        positions=positions,
        atom_mask=atom_mask,
        asym_id=asym_id,
        interface_threshold=interface_threshold,
    )

    if interface_residues.numel() == 0:
        return _get_contiguous_crop_idx(
            asym_id=asym_id,
            seq_length=positions.shape[0],
            crop_size=crop_size,
            generator=generator,
        )

    target_res_idx = _randint(
        lower=0,
        upper=interface_residues.shape[-1] - 1,
        generator=generator,
        device=positions.device,
    )
    target_res = interface_residues[target_res_idx]

    ca_idx = residue_constants.atom_order["CA"]
    ca_positions = positions[..., ca_idx, :]
    ca_mask = atom_mask[..., ca_idx].bool()

    coord_diff = ca_positions[..., None, :] - ca_positions[..., None, :, :]
    ca_pairwise_dists = torch.sqrt(torch.sum(coord_diff ** 2, dim=-1))
    to_target_distances = ca_pairwise_dists[target_res]
    break_tie = (
        torch.arange(0, to_target_distances.shape[-1], device=positions.device).float()
        * 1e-3
    )
    to_target_distances = torch.where(ca_mask, to_target_distances, torch.inf) + break_tie
    crop_idxs = torch.argsort(to_target_distances)[:crop_size]
    return crop_idxs.sort().values


def _select_multimer_crop_idx(chain_feats: dict, data_conf, asym_id_1d=None, generator=None):
    crop_conf = getattr(data_conf, 'crop', None)
    if crop_conf is None or not bool(getattr(crop_conf, 'enabled', False)):
        return None
    crop_size = int(getattr(crop_conf, 'crop_size', 0) or 0)
    if crop_size <= 0:
        return None

    spatial_crop_prob = float(getattr(crop_conf, 'spatial_crop_prob', 0.5))
    interface_threshold = float(getattr(crop_conf, 'interface_threshold', 8.0))

    # Crop is selected from motion coordinates (if available) or atom37_pos
    if 'motion_atom37_pos' in chain_feats:
        positions = torch.as_tensor(chain_feats['motion_atom37_pos'][-1]).float()
    else:
        # Use atom37_pos for seq->structure model (motion_number=0)
        positions = torch.as_tensor(chain_feats['atom37_pos'][-1]).float()
    atom_mask = torch.as_tensor(chain_feats['atom37_mask'][0]).float()
    num_res = positions.shape[0]
    if num_res <= crop_size:
        return None

    if asym_id_1d is None:
        asym_id = torch.ones((num_res,), dtype=torch.long, device=positions.device)
    else:
        asym_id = torch.as_tensor(asym_id_1d, dtype=torch.long, device=positions.device)
        if asym_id.shape[0] != num_res:
            raise ValueError(
                f"asym_id length mismatch for crop: asym_id={asym_id.shape[0]} vs num_res={num_res}"
            )

    use_spatial_crop = (
        torch.rand((1,), device=positions.device, generator=generator) < spatial_crop_prob
    )
    if use_spatial_crop:
        return _get_spatial_crop_idx(
            positions=positions,
            atom_mask=atom_mask,
            asym_id=asym_id,
            crop_size=crop_size,
            interface_threshold=interface_threshold,
            generator=generator,
        )
    return _get_contiguous_crop_idx(
        asym_id=asym_id,
        seq_length=num_res,
        crop_size=crop_size,
        generator=generator,
    )

def parse_dynamics_chain_feats(chain_feats, scale_factor=1.):
    ca_idx = residue_constants.atom_order['CA']
    chain_feats['bb_mask'] = chain_feats['all_atom_mask'][:, ca_idx] # [N,37]
    bb_pos = chain_feats['all_atom_positions'][0, :, ca_idx] # [F,N,37,3]->[N,3] select first protein as anchor
    bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats['bb_mask']) + 1e-5) # [3]
    centered_pos = chain_feats['all_atom_positions'] - bb_center[None, None, None, :] # [F,N,37,3]
    scaled_pos = centered_pos / scale_factor
    chain_feats['all_atom_positions'] = scaled_pos * (chain_feats['all_atom_mask'][..., None][np.newaxis, ...])
    chain_feats['bb_positions'] = chain_feats['all_atom_positions'][:, :, ca_idx]# [F,N,3]
    return chain_feats


class PdbDataset(data.Dataset):
    def __init__(
            self,
            *,
            data_conf,
            diffuser,
            is_training,
            is_testing=False,
            is_random_test=False,
            rank=0, # only used for multi node evaluation
            grouped=1 # only used for multi node evaluation
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._is_test = is_testing
        self._is_random_test=is_random_test
        self._data_conf = data_conf
        self._init_metadata(rank, grouped)
        self._diffuser = diffuser
        self.offset =  {idx: 0 for idx in range(len(self.csv))}

    @property
    def is_training(self):
        return self._is_training

    @property
    def diffuser(self):
        return self._diffuser

    @property
    def data_conf(self):
        return self._data_conf

    def _init_metadata(self, rank=0, grouped=1):
        """Initialize metadata."""
        def _len_col(df: pd.DataFrame) -> str:
            return "total_seq_len" if "total_seq_len" in df.columns else "seq_len"

        filter_conf = self.data_conf.filtering
        if self._is_training:
            pdb_csv = pd.read_csv(self.data_conf.csv_path)
            before = len(pdb_csv)
            pdb_csv = pdb_csv.dropna()
            after = len(pdb_csv)
            if before != after:
                print(f"[INFO] Dropped {before - after} rows containing None/NaN values (kept {after} rows).")
            lcol = _len_col(pdb_csv)
            crop_conf = getattr(self.data_conf, "crop", None)
            crop_enabled = crop_conf is not None and bool(getattr(crop_conf, "enabled", False))

            if crop_enabled:
                # 开启 crop 时，不在 metadata 阶段按 max_len 丢弃长样本
                filter_len = len(pdb_csv)
                print(f"[INFO] Crop enabled: skip length filtering in metadata (kept {filter_len} rows).")
            else:
                pdb_csv = pdb_csv[pdb_csv[lcol] <= filter_conf.train_max_len]
                filter_len = len(pdb_csv)
                print(f"[INFO] Dropped {after - filter_len} rows exceeding len (kept {filter_len} rows).")
        elif self._is_test:
            pdb_csv = pd.read_csv(self.data_conf.test_csv_path)
            before = len(pdb_csv)
            pdb_csv = pdb_csv.dropna()
            after = len(pdb_csv)
            if before != after:
                print(f"[INFO] Dropped {before - after} rows containing None/NaN values (kept {after} rows).")

            # Apply the same length filtering for test.
            lcol = _len_col(pdb_csv)
            pdb_csv = pdb_csv[pdb_csv[lcol] <= filter_conf.test_max_len]
            filter_len = len(pdb_csv)
            print(f"[INFO] Test: Dropped {after - filter_len} rows exceeding len (kept {filter_len} rows).")

            pdb_csv = pdb_csv[rank::grouped]
            print(f"Grouped by {grouped} | Rank: {rank}")
            pdb_csv = pdb_csv.head(self.data_conf.max_protein_num)
        else:
            pdb_csv = pd.read_csv(self.data_conf.val_csv_path)
            before = len(pdb_csv)
            pdb_csv = pdb_csv.dropna()
            after = len(pdb_csv)
            if before != after:
                print(f"[INFO] Dropped {before - after} rows containing None/NaN values (kept {after} rows).")

            # Apply the same length filtering for validation.
            lcol = _len_col(pdb_csv)
            pdb_csv = pdb_csv[pdb_csv[lcol] <= filter_conf.val_max_len]
            filter_len = len(pdb_csv)
            print(f"[INFO] Validation: Dropped {after - filter_len} rows exceeding len (kept {filter_len} rows).")
        print(pdb_csv)
        self._create_split(pdb_csv)

    def _create_split(self, pdb_csv):
        # Training or validation specific logic.
        if self.is_training:
            self.csv = pdb_csv#[pdb_csv.split == 'train']
            self._log.info(f'Training: {len(self.csv)} examples')
        else:
            self.csv = pdb_csv#[pdb_csv.split == 'val']
            self._log.info(f'Validation: {len(self.csv)} examples')

    def select_random_samples(self,arr, t, k):
        n = arr.shape[0]  # Obtain the size of the first dimension, the number of samples
        if t > n:
            raise ValueError("t cannot be greater than the number of samples")
        start_index = np.random.randint(0, n - (t)*k + 1)  # randomly select the start indexnp.random.randint(0, n - t*(k-1))
        end_index = start_index + (t)*k # the end index
        selected_samples = arr[start_index:end_index:k]  # select with step k
        return selected_samples,start_index


    def select_first_samples(self, arr, number, k):
        n = arr.shape[0]  # length of trajectory
        if number > n:
            raise ValueError("t cannot be greater than the number of samples")

        start_index = 0 #np.random.randint(0, n - (t)*k + 1)  # Randomly select the starting index.
        end_index = start_index + (number) * k # endding index
        selected_samples = arr[start_index:end_index:k]  # Select t consecutive samples with a step of k.
        return selected_samples


    def select_with_motion_continue(self, arr, number, k, ref_frame):
        arr = np.asarray(arr)
        n = len(arr)
        max_start_index = n - number * k - 1
        if max_start_index < 0:
            raise ValueError(
                "The array is too small to select t elements with the given interval s."
            )
        start_index = np.random.randint(0, max_start_index + 1)
        motion_part = arr[start_index : start_index + number * k : k]
        ref_indices = np.random.choice(n, size=ref_frame, replace=False)
        ref_part = arr[ref_indices]
        combined = np.concatenate([ref_part, motion_part])
        return combined

    def _process_csv_row(self, processed_file_path):
        processed_feats = dict(_np_load_allow_pickle_compat(processed_file_path))

        motion_frame = self.data_conf.motion_number
        ref_frame = self.data_conf.ref_number
        frame_time = self.data_conf.frame_time
        # here to sample frame_time continuous positions.
        frame_time_ref_motion = ref_frame + motion_frame + frame_time
        motion_frame_number = motion_frame + frame_time
        
        # Detect data format: trajectory [T, N, 37, 3] vs structure-only [N, 37, 3]
        all_atom_pos = processed_feats["all_atom_positions"]
        is_structure_only = (all_atom_pos.ndim == 3)  # [N, 37, 3]
        
        if is_structure_only:
            # Structure-only format: add frame dimension [1, N, 37, 3]
            processed_feats["all_atom_positions"] = all_atom_pos[np.newaxis, ...]
        
        if self._is_training:
            train_source = processed_feats["all_atom_positions"]
            keep_first = getattr(self.data_conf, "keep_first", None)
            if keep_first is not None:
                train_source = train_source[:keep_first]
            train_reference_mode = getattr(self.data_conf, "train_reference_mode", "contiguous")
            if train_reference_mode == "contiguous":
                tmp, _ = self.select_random_samples(
                    train_source,
                    frame_time_ref_motion,
                    self.data_conf.frame_sample_step,
                )
            elif train_reference_mode == "random_ref":
                tmp = self.select_with_motion_continue(
                    train_source,
                    motion_frame_number,
                    self.data_conf.frame_sample_step,
                    ref_frame,
                )
            else:
                raise ValueError(
                    f"Unsupported train_reference_mode={train_reference_mode!r}. "
                    "Expected one of {'contiguous', 'random_ref'}."
                )

        else:
            tmp = self.select_first_samples(
                processed_feats["all_atom_positions"],
                number=frame_time_ref_motion,
                k=self.data_conf.frame_sample_step,
            )
            start_index = 0

        processed_feats['all_atom_positions'] = tmp
        processed_feats = parse_dynamics_chain_feats(processed_feats)
        
        # Get aatype indices - handle both one-hot and index formats
        if processed_feats['aatype'].ndim == 1:
            # Already indices format [N]
            aatype_indices = processed_feats['aatype'].astype(int)
        else:
            # One-hot encoded format [N, num_types]
            aatype_indices = np.argmax(processed_feats['aatype'], axis=-1)
        
        # Determine RNA from aatype indices
        # aatype can be either Protenix encoding (21-25 for RNA) or internal (0-3 for RNA)
        # Protenix: 21=A, 22=G, 23=C, 24=U, 25=N
        # Internal: 0=A, 1=G, 2=C, 3=U (if using one-hot with 4 classes)
        
        # Get number of residue types based on aatype format
        if processed_feats['aatype'].ndim == 1:
            num_residue_types = None  # Will be inferred from data
        else:
            num_residue_types = processed_feats['aatype'].shape[-1]
        
        if 'is_rna' in processed_feats:
            # Use explicit RNA flag if available
            is_rna_np = processed_feats['is_rna'].astype(bool)
        elif num_residue_types == 4:
            # Pure RNA data with 4-class one-hot (A, G, C, U)
            is_rna_np = np.ones(len(aatype_indices), dtype=bool)
        elif num_residue_types is None:
            # Index format - detect RNA by Protenix indices
            is_rna_np = np.array([is_rna_residue(int(a)) for a in aatype_indices])
        else:
            # Mixed or protein data - detect RNA by Protenix indices
            is_rna_np = np.array([is_rna_residue(int(a)) for a in aatype_indices])
        
        # Create Protenix-compatible restype encoding
        if 'restype_protenix' in processed_feats:
            # Use pre-computed Protenix encoding
            restype_protenix = processed_feats['restype_protenix']
        elif num_residue_types == 4 and is_rna_np.all():
            # Pure RNA with internal 0-3 encoding -> map to Protenix 21-24
            # Internal: 0=A, 1=G, 2=C, 3=U
            # Protenix: 21=A, 22=G, 23=C, 24=U
            restype_protenix = aatype_indices + 21
        elif is_rna_np.any():
            # Mixed data - already has Protenix encoding for RNA, protein is 0-19
            restype_protenix = aatype_indices.copy()
        else:
            # Pure protein data
            restype_protenix = aatype_indices.copy()
        
        chain_feats = {
            'aatype': torch.tensor(restype_protenix).long().unsqueeze(0).expand(frame_time_ref_motion, -1),
            'is_rna': torch.tensor(is_rna_np, dtype=torch.bool).unsqueeze(0).expand(frame_time_ref_motion, -1),
            'restype_protenix': torch.tensor(restype_protenix).long().unsqueeze(0).expand(frame_time_ref_motion, -1),
            'all_atom_positions': torch.tensor(processed_feats['all_atom_positions']).double(),
            'all_atom_mask': torch.tensor(processed_feats['all_atom_mask']).double().unsqueeze(0).expand(frame_time_ref_motion, -1, -1)
        }

        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)
        
        #NOTE motion ref frame - only create if motion_frame > 0 (seq->structure model compatibility)
        motion_feats = {}
        if motion_frame > 0:
            motion_feats = {
                'motion_rigids_0': rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][ref_frame: ref_frame+motion_frame])[:, :, 0].to_tensor_7(),
                'motion_node_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(motion_frame, -1),
                'motion_atom37_pos': chain_feats['all_atom_positions'][ref_frame: ref_frame+motion_frame],
            }
        
        #NOTE ref frame - only create if ref_frame > 0 (seq->structure model compatibility)
        ref_feats = {}
        if ref_frame > 0:
            ref_feats = {
                'ref_rigids_0': rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][:ref_frame])[:, :, 0].to_tensor_7(),
                'ref_node_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(ref_frame, -1),
                'ref_atom37_pos': chain_feats['all_atom_positions'][:ref_frame],
            }

        # Propagate is_rna flag to final features
        is_rna_tensor = torch.tensor(is_rna_np, dtype=torch.bool).unsqueeze(0).expand(frame_time, -1)
        
        final_feats = {
            'aatype': chain_feats['aatype'][ref_frame+motion_frame:],
            'is_rna': is_rna_tensor,
            'seq_idx':  torch.tensor(processed_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            # 'chain_idx': new_chain_idx,
            'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'][ref_frame+motion_frame:],
            'residue_index': torch.tensor(processed_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            'res_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(frame_time, -1),
            'atom37_pos': chain_feats['all_atom_positions'][ref_frame+motion_frame:],
            'atom37_mask': chain_feats['all_atom_mask'][ref_frame+motion_frame:],
            # 'atom14_pos': chain_feats['atom14_gt_positions'],
            'rigidgroups_0': chain_feats['rigidgroups_gt_frames'][ref_frame+motion_frame:],
            'torsion_angles_sin_cos': chain_feats['torsion_angles_sin_cos'][ref_frame+motion_frame:],
            'alt_torsion_angles_sin_cos':chain_feats['alt_torsion_angles_sin_cos'][ref_frame+motion_frame:],
            'torsion_angles_mask':chain_feats['torsion_angles_mask'][ref_frame+motion_frame:],
        }

        # Only add ref/motion feats if they exist (seq->structure model compatibility)
        if ref_feats:
            final_feats.update(ref_feats)
        if motion_feats:
            final_feats.update(motion_feats)

        if not self._is_training:
            final_feats.update({'start_index':start_index})

        return final_feats

    def _create_diffused_masks(self, atom37_pos, rng, row):
        bb_pos = atom37_pos[:, residue_constants.atom_order['CA']]
        dist2d = np.linalg.norm(bb_pos[:, None, :] - bb_pos[None, :, :], axis=-1)

        # Randomly select residue then sample a distance cutoff
        # TODO: Use a more robust diffuse mask sampling method.
        diff_mask = np.zeros_like(bb_pos)
        attempts = 0
        while np.sum(diff_mask) < 1:
            crop_seed = rng.integers(dist2d.shape[0])
            seed_dists = dist2d[crop_seed]
            max_scaffold_size = min(
                self._data_conf.scaffold_size_max,
                seed_dists.shape[0] - self._data_conf.motif_size_min
            )
            scaffold_size = rng.integers(
                low=self._data_conf.scaffold_size_min,
                high=max_scaffold_size
            )
            dist_cutoff = np.sort(seed_dists)[scaffold_size]
            diff_mask = (seed_dists < dist_cutoff).astype(float)
            attempts += 1
            if attempts > 100:
                raise ValueError(
                    f'Unable to generate diffusion mask for {row}')
        return diff_mask

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        # Sample data example.
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        if "pdb_id" in csv_row:
            pdb_name = csv_row['pdb_id']
        else:
            raise ValueError('Need chain identifier.')
        processed_file_path = csv_row["pos_path"]
        chain_feats = self._process_csv_row(processed_file_path)


        frame_time = chain_feats['aatype'].shape[0]
        node_edge_feature_path = csv_row['embed_path']  # here
        assert os.path.exists(node_edge_feature_path)
        attr_dict = dict(np.load(node_edge_feature_path))

        # ---- PairFormer residue-level feature loading (multimer-ready) ----
        # Support keys:
        #   - single_s (N,384), pair_z (N,N,128)   [your current .pairformer.residue.npz]
        #   - node_repr/edge_repr                 [older naming]
        #   - single/pair                         [older naming]
        if "single_s" in attr_dict:
            node = attr_dict["single_s"]
        elif "node_repr" in attr_dict:
            node = attr_dict["node_repr"]
        elif "single" in attr_dict:
            node = attr_dict["single"]
        else:
            raise KeyError(f"Cannot find node feature in npz keys={list(attr_dict.keys())}")

        if "pair_z" in attr_dict:
            edge = attr_dict["pair_z"]
        elif "edge_repr" in attr_dict:
            edge = attr_dict["edge_repr"]
        elif "pair" in attr_dict:
            edge = attr_dict["pair"]
        else:
            raise KeyError(f"Cannot find edge feature in npz keys={list(attr_dict.keys())}")

        # Load embeddings and normalize to prevent numerical instability
        # Protenix embeddings can have very large values (up to 3000+) which cause NaN gradients
        node_tensor = torch.tensor(node, dtype=torch.float32)
        edge_tensor = torch.tensor(edge, dtype=torch.float32)
        
        # Always normalize Protenix embeddings to unit variance
        # This is critical for numerical stability in the IPA modules
        node_std = node_tensor.std()
        if node_std > 1.0:
            node_tensor = node_tensor / node_std
        
        edge_std = edge_tensor.std()
        if edge_std > 1.0:
            edge_tensor = edge_tensor / edge_std
        
        chain_feats["node_repr"] = node_tensor
        chain_feats["edge_repr"] = edge_tensor
        asym_for_crop = None
        if "asym_id" in attr_dict:
            asym_for_crop = torch.tensor(
                np.asarray(attr_dict["asym_id"]).astype(np.int64)
            ).long()
            if asym_for_crop.shape[0] != chain_feats["node_repr"].shape[0]:
                raise ValueError(
                    f"Residue length mismatch for crop: node_repr N={chain_feats['node_repr'].shape[0]} "
                    f"vs asym_id N={asym_for_crop.shape[0]}"
                )
            # Keep asym_id in batch for multimer losses.
            chain_feats["asym_id"] = asym_for_crop.unsqueeze(0).expand(frame_time, -1)
            if not self.is_training:
                pdb_chain_index = _remap_asym_id_to_pdb_chain_index(asym_for_crop)
                chain_feats["pdb_chain_index"] = pdb_chain_index
                if "residue_index" in attr_dict:
                    pdb_residue_index = torch.tensor(
                        np.asarray(attr_dict["residue_index"]).astype(np.int64)
                    ).long()
                else:
                    pdb_residue_index = chain_feats["residue_index"][0].long()
                chain_feats["pdb_residue_index"] = pdb_residue_index

        # === AlphaFold-Multimer crop (spatial / contiguous) ===
        if self.is_training:
            crop_idx = _select_multimer_crop_idx(
                chain_feats,
                self._data_conf,
                asym_id_1d=asym_for_crop,
            )
            if crop_idx is not None:
                chain_feats = _crop_feature_dict_by_residue_idx(
                    chain_feats,
                    crop_idx.long(),
                    original_n_res=int(chain_feats["res_mask"].shape[-1]),
                )

                # Sanity check after crop
                n_res = int(chain_feats["res_mask"].shape[-1])
                if chain_feats["node_repr"].shape[0] != n_res:
                    raise ValueError(
                        f"node_repr length mismatch after crop: "
                        f"{chain_feats['node_repr'].shape[0]} vs res_mask {n_res}"
                    )
                if chain_feats["edge_repr"].shape[0] != n_res or chain_feats["edge_repr"].shape[1] != n_res:
                    raise ValueError(
                        f"edge_repr shape mismatch after crop: "
                        f"{tuple(chain_feats['edge_repr'].shape)} vs N={n_res}"
                    )
                if "asym_id" in chain_feats and chain_feats["asym_id"].shape[-1] != n_res:
                    raise ValueError(
                        f"asym_id length mismatch after crop: "
                        f"{chain_feats['asym_id'].shape[-1]} vs N={n_res}"
                    )

        #TODO prob to be zero reference pos information
        # Only apply if ref features exist (seq->structure model compatibility)
        if random.random() < self._data_conf.cfg_drop_rate and 'ref_rigids_0' in chain_feats:
            chain_feats['ref_rigids_0'] = torch.zeros_like(chain_feats['ref_rigids_0'])
            chain_feats['ref_atom37_pos'] = torch.zeros_like(chain_feats['ref_atom37_pos'])
        # Use a fixed seed for evaluation.
        if self.is_training:
            rng = np.random.default_rng(None)
        else:
            rng = np.random.default_rng(idx)

        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_0'])[:, :, 0]
        diffused_mask = np.ones_like(chain_feats['res_mask'])
        if np.sum(diffused_mask) < 1:
            raise ValueError('Must be diffused')
        fixed_mask = 1 - diffused_mask
        chain_feats['fixed_mask'] = fixed_mask
        chain_feats['rigids_0'] = gt_bb_rigid.to_tensor_7()
        chain_feats['sc_ca_t'] = torch.zeros_like(gt_bb_rigid.get_trans())
        # Sample t and diffuse.
        if self.is_training:
            # TODO here add different t
            t = rng.uniform(self._data_conf.min_t, 1.0)
            diff_feats_t = self._diffuser.forward_marginal(
                rigids_0=gt_bb_rigid,
                t=t,
                diffuse_mask=None
            )
            # Only compute ref diffusion if ref features exist (seq->structure model compatibility)
            if 'ref_rigids_0' in chain_feats:
                ref_diff_feats = self._diffuser.forward_marginal(
                    rigids_0=rigid_utils.Rigid.from_tensor_7(chain_feats['ref_rigids_0']),
                    t=t,
                    diffuse_mask=None
                )
                chain_feats['ref_rot_score'] = ref_diff_feats['rot_score']
        else:
            t = 1.0
            # todo here
            if self._data_conf.dynamics:
                diff_feats_t = self.diffuser.sample_ref(
                    n_samples=gt_bb_rigid.shape[0]*gt_bb_rigid.shape[1],
                    diffuse_mask=None,
                    as_tensor_7=True,
                )
            else:
                diff_feats_t = self.diffuser.sample_ref(
                    n_samples=gt_bb_rigid.shape[0],
                    impute=gt_bb_rigid,
                    diffuse_mask=None,
                    as_tensor_7=True,
                )

        chain_feats.update(diff_feats_t)

        chain_feats['t'] = t
        if not self.is_training:
            start_index = chain_feats.pop('start_index')
        # Convert all features to tensors.
        final_feats = tree.map_structure(lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats)

        if self.is_training:
            return final_feats
        else:
            return final_feats, pdb_name, start_index


    def _get_row(self, idx):
        # Sample data example.
        csv_row = self.csv.iloc[idx]
        # if "pdb_id" in csv_row:
        #     pdb_name = csv_row["pdb_id"]
        # else:
        #     raise ValueError("Need chain identifier.")

        processed_file_path = csv_row["pos_path"]
        pdb_name = os.path.basename(processed_file_path).split(".")[0]
        # chain_feats = self._process_csv_row(processed_file_path)

        processed_feats = dict(_np_load_allow_pickle_compat(processed_file_path))
        # process feats
        motion_frame = self.data_conf.motion_number
        ref_frame = self.data_conf.ref_number
        frame_time = self.data_conf.frame_time
        # here to sample frame_time continuous positions.
        # frame_time_ref_motion = ref_frame + motion_frame + frame_time
        frame_time_ref_motion = ref_frame + motion_frame
        tmp = self.select_first_samples(
            processed_feats["all_atom_positions"] if not self.data_conf.last else processed_feats["all_atom_positions"][-frame_time_ref_motion:],
            number=frame_time_ref_motion,
            k=self.data_conf.frame_sample_step,
        )

        processed_feats['all_atom_positions'] = tmp
        processed_feats = parse_dynamics_chain_feats(processed_feats)
        chain_feats = {
            'aatype': torch.tensor(np.argmax(processed_feats['aatype'],axis=-1)).long().unsqueeze(0).expand(frame_time_ref_motion, -1),
            'is_rna': torch.tensor([is_rna_residue(int(a)) for a in np.argmax(processed_feats['aatype'],axis=-1)], dtype=torch.bool).unsqueeze(0).expand(frame_time_ref_motion, -1),
            'all_atom_positions': torch.tensor(processed_feats['all_atom_positions']).double(),
            'all_atom_mask': torch.tensor(processed_feats['all_atom_mask']).double().unsqueeze(0).expand(frame_time_ref_motion, -1, -1)
        }

        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)
        
        # motion ref frame - only create if motion_frame > 0 (seq->structure model compatibility)
        motion_feats = {}
        if motion_frame > 0:
            motion_feats = {
                'motion_rigids_0': rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][ref_frame: ref_frame+motion_frame])[:,:, 0].to_tensor_7(),
                'motion_node_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(motion_frame, -1),
                'motion_atom37_pos': chain_feats['all_atom_positions'][ref_frame: ref_frame+motion_frame],
            }
        
        # ref frame - only create if ref_frame > 0 (seq->structure model compatibility)
        ref_feats = {}
        if ref_frame > 0:
            ref_feats = {
                'ref_rigids_0': rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][:ref_frame])[:,:, 0].to_tensor_7(),
                'ref_node_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(ref_frame, -1),
                'ref_atom37_pos': chain_feats['all_atom_positions'][:ref_frame],
            }

        final_feats = {
            'aatype': chain_feats['aatype'][:1].expand(frame_time, -1),
            'seq_idx':  torch.tensor(processed_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            # 'chain_idx': new_chain_idx,
            'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'][:1].expand(frame_time, -1, -1),
            'residue_index': torch.tensor(processed_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            'res_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(frame_time, -1),
            'atom37_pos': chain_feats['all_atom_positions'][:1].expand(frame_time, -1, -1, -1),
            'atom37_mask': chain_feats['all_atom_mask'][:1].expand(frame_time, -1, -1),
            # 'atom14_pos': chain_feats['atom14_gt_positions'],
            'rigidgroups_0': chain_feats['rigidgroups_gt_frames'][:1].expand(frame_time, -1,-1,-1,-1),
            'torsion_angles_sin_cos': chain_feats['torsion_angles_sin_cos'][:1].expand(frame_time, -1,-1,-1),
            'alt_torsion_angles_sin_cos':chain_feats['alt_torsion_angles_sin_cos'][:1].expand(frame_time, -1,-1,-1),
            'torsion_angles_mask':chain_feats['torsion_angles_mask'][:1].expand(frame_time, -1,-1),
        }

        # Only add ref/motion feats if they exist (seq->structure model compatibility)
        if ref_feats:
            final_feats.update(ref_feats)
        if motion_feats:
            final_feats.update(motion_feats)
        chain_feats = final_feats
        # process feats

        node_edge_feature_path = csv_row["embed_path"]  # here
        assert os.path.exists(node_edge_feature_path)
        attr_dict = dict(np.load(node_edge_feature_path))

        # ---- PairFormer residue-level feature loading (multimer-ready) ----
        # Support keys:
        #   - single_s (N,384), pair_z (N,N,128)   [your current .pairformer.residue.npz]
        #   - node_repr/edge_repr                 [older naming]
        #   - single/pair                         [older naming]
        if "single_s" in attr_dict:
            node = attr_dict["single_s"]
        elif "node_repr" in attr_dict:
            node = attr_dict["node_repr"]
        elif "single" in attr_dict:
            node = attr_dict["single"]
        else:
            raise KeyError(f"Cannot find node feature in npz keys={list(attr_dict.keys())}")

        if "pair_z" in attr_dict:
            edge = attr_dict["pair_z"]
        elif "edge_repr" in attr_dict:
            edge = attr_dict["edge_repr"]
        elif "pair" in attr_dict:
            edge = attr_dict["pair"]
        else:
            raise KeyError(f"Cannot find edge feature in npz keys={list(attr_dict.keys())}")

        # Load embeddings and normalize to prevent numerical instability
        # Protenix embeddings can have very large values (up to 3000+) which cause NaN gradients
        node_tensor = torch.tensor(node, dtype=torch.float32)
        edge_tensor = torch.tensor(edge, dtype=torch.float32)
        
        # Always normalize Protenix embeddings to unit variance
        # This is critical for numerical stability in the IPA modules
        node_std = node_tensor.std()
        if node_std > 1.0:
            node_tensor = node_tensor / node_std
        
        edge_std = edge_tensor.std()
        if edge_std > 1.0:
            edge_tensor = edge_tensor / edge_std
        
        chain_feats["node_repr"] = node_tensor
        chain_feats["edge_repr"] = edge_tensor
        
        if "asym_id" in attr_dict:
            asym = torch.tensor(np.asarray(attr_dict["asym_id"]).astype(np.int64)).long()
            if asym.shape[0] != chain_feats["node_repr"].shape[0]:
                raise ValueError(
                    f"Residue length mismatch: node_repr N={chain_feats['node_repr'].shape[0]} "
                    f"vs asym_id N={asym.shape[0]}"
                )
            chain_feats["asym_id"] = asym.unsqueeze(0).expand(frame_time, -1)
            pdb_chain_index = _remap_asym_id_to_pdb_chain_index(asym)
            chain_feats["pdb_chain_index"] = pdb_chain_index
            if "residue_index" in attr_dict:
                pdb_residue_index = torch.tensor(
                    np.asarray(attr_dict["residue_index"]).astype(np.int64)
                ).long()
            else:
                pdb_residue_index = chain_feats["residue_index"][0].long()
            chain_feats["pdb_residue_index"] = pdb_residue_index

        # Use a fixed seed for evaluation.
        rng = np.random.default_rng(idx)

        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats["rigidgroups_0"])[:, :, 0]
        diffused_mask = np.ones_like(chain_feats["res_mask"])
        if np.sum(diffused_mask) < 1:
            raise ValueError("Must be diffused")
        fixed_mask = 1 - diffused_mask
        chain_feats["fixed_mask"] = fixed_mask
        chain_feats["rigids_0"] = gt_bb_rigid.to_tensor_7()
        chain_feats["sc_ca_t"] = torch.zeros_like(gt_bb_rigid.get_trans())

        t = 1.0
        # todo here
        if self._data_conf.dynamics:
            diff_feats_t = self.diffuser.sample_ref(
                n_samples=frame_time * gt_bb_rigid.shape[1],
                diffuse_mask=None,
                as_tensor_7=True,
            )
        else:
            diff_feats_t = self.diffuser.sample_ref(
                n_samples=frame_time,
                impute=gt_bb_rigid,
                diffuse_mask=None,
                as_tensor_7=True,
            )

        chain_feats.update(diff_feats_t)

        chain_feats["t"] = t
        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats
        )
        return final_feats, pdb_name
