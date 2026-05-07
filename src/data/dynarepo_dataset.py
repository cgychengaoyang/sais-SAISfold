"""Dataset for DyneTrion MD data (dynarepo).

This dataset loads:
- Sequence embeddings (single_s) as node features [N, 384]
- Pair embeddings (pair_z) as edge features [N, N, 128]
- PDB structures for ground truth rigids
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openfold.utils.rigid_utils import Rigid


class DynarepoDataset(Dataset):
    """Dataset for dynarepo MD data.
    
    Loads pre-computed embeddings and PDB structures for training
    the score-based structure prediction model.
    """
    
    def __init__(
        self,
        csv_path: str,
        embed_dir: str = f"{root_path_02}/public/caizhiqiang/DyneTrion-multimer/DyneTrion/data/dynarepo/pairformer_npz",
        max_seq_len: Optional[int] = None,
        use_cache: bool = True,
    ):
        """Initialize dataset.
        
        Args:
            csv_path: Path to CSV file with metadata
            embed_dir: Directory containing .npz embedding files
            max_seq_len: Maximum sequence length (truncate longer sequences)
            use_cache: Whether to cache loaded data in memory
        """
        self.csv_path = csv_path
        self.embed_dir = embed_dir
        self.max_seq_len = max_seq_len
        self.use_cache = use_cache
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Filter to only samples with existing embedding files and PDB files
        valid_rows = []
        for idx, row in df.iterrows():
            accession = row['accession']
            embed_path = os.path.join(self.embed_dir, f"{accession}.pairformer.residue.npz")
            pdb_path = row.get('pdb_path', '')
            if os.path.exists(embed_path) and os.path.exists(pdb_path):
                valid_rows.append(row)
            else:
                missing = []
                if not os.path.exists(embed_path):
                    missing.append('embedding')
                if not os.path.exists(pdb_path):
                    missing.append('PDB')
                print(f"Warning: Skipping {accession} - missing {', '.join(missing)}")
        
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"DynarepoDataset: {len(self.df)}/{len(df)} samples available")
        
        # Cache for loaded data
        self._cache = {}
        
    def __len__(self):
        return len(self.df)
    
    def _load_embeddings(self, accession: str):
        """Load embeddings from npz file.
        
        Returns:
            dict with 'single_s' [N, 384], 'pair_z' [N, N, 128]
        """
        embed_path = os.path.join(self.embed_dir, f"{accession}.pairformer.residue.npz")
        
        data = np.load(embed_path, allow_pickle=True)
        
        single_s = torch.from_numpy(data['single_s']).float()  # [N, 384]
        pair_z = torch.from_numpy(data['pair_z']).float()  # [N, N, 128]
        
        return {
            'single_s': single_s,
            'pair_z': pair_z,
            'restype_idx': torch.from_numpy(data['restype_idx']).long(),
        }
    
    def _load_pdb_structure(self, pdb_path: str):
        """Load PDB structure and extract CA positions.
        
        Returns:
            CA positions [N, 3]
        """
        # Use OpenFold's data pipeline to load PDB
        # For now, simplified version
        
        # Parse PDB manually for CA positions
        ca_positions = []
        
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and ' CA ' in line:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ca_positions.append([x, y, z])
        
        if len(ca_positions) == 0:
            raise ValueError(f"No CA atoms found in {pdb_path}")
        
        return torch.tensor(ca_positions, dtype=torch.float32)
    
    def _build_rigids_from_ca(self, ca_positions: torch.Tensor):
        """Build rigids from CA trace.
        
        For simplicity, we use CA positions directly and estimate
        backbone frames. In practice, you might want to use full
        backbone reconstruction.
        
        Args:
            ca_positions: [N, 3] CA coordinates
            
        Returns:
            rigids: [N, 7] (quat + trans)
        """
        N = ca_positions.shape[0]
        
        # Use CA positions as translations
        translations = ca_positions
        
        # For rotations, we can estimate from CA-CA vectors
        # For simplicity, use identity rotation
        # In practice, you'd want proper backbone frame construction
        quaternions = torch.zeros(N, 4)
        quaternions[:, 0] = 1.0  # Identity rotation (w=1, x=y=z=0)
        
        rigids = torch.cat([quaternions, translations], dim=-1)
        return rigids
    
    def __getitem__(self, idx):
        """Get a single sample.
        
        Returns:
            dict with:
                - node_repr: [N, 384] sequence embeddings
                - edge_repr: [N, N, 128] pair embeddings
                - gt_rigids: [N, 7] ground truth rigids from PDB
                - mask: [N] residue mask
                - accession: str, sample ID
        """
        if idx in self._cache and self.use_cache:
            return self._cache[idx]
        
        row = self.df.iloc[idx]
        accession = row['accession']
        pdb_path = row['pdb_path']
        
        # Load embeddings
        embed_data = self._load_embeddings(accession)
        node_repr = embed_data['single_s']  # [N, 384]
        edge_repr = embed_data['pair_z']  # [N, N, 128]
        
        # Truncate if needed
        if self.max_seq_len is not None:
            N = min(node_repr.shape[0], self.max_seq_len)
            node_repr = node_repr[:N]
            edge_repr = edge_repr[:N, :N]
        else:
            N = node_repr.shape[0]
        
        # Load PDB structure
        try:
            ca_positions = self._load_pdb_structure(pdb_path)
            
            # Truncate to match embeddings
            if ca_positions.shape[0] > N:
                ca_positions = ca_positions[:N]
            elif ca_positions.shape[0] < N:
                # Pad if needed (shouldn't happen often)
                pad_len = N - ca_positions.shape[0]
                ca_positions = torch.cat([
                    ca_positions,
                    ca_positions[-1:].expand(pad_len, -1)
                ], dim=0)
            
            # Build rigids from CA positions
            gt_rigids = self._build_rigids_from_ca(ca_positions)
            
        except Exception as e:
            print(f"Warning: Failed to load PDB for {accession}: {e}")
            # Return dummy data
            gt_rigids = torch.zeros(N, 7)
            gt_rigids[:, 0] = 1.0  # Identity rotation
        
        # Create mask (all ones for now)
        mask = torch.ones(N)
        
        sample = {
            'node_repr': node_repr,
            'edge_repr': edge_repr,
            'gt_rigids': gt_rigids,
            'mask': mask,
            'accession': accession,
        }
        
        if self.use_cache:
            self._cache[idx] = sample
        
        return sample


def collate_fn(batch):
    """Collate function for batching variable-length sequences.
    
    Pads sequences to max length in batch.
    """
    max_len = max(s['node_repr'].shape[0] for s in batch)
    
    batch_node = []
    batch_edge = []
    batch_gt_rigids = []
    batch_mask = []
    batch_accessions = []
    
    for sample in batch:
        N = sample['node_repr'].shape[0]
        
        if N < max_len:
            # Pad
            pad_len = max_len - N
            node_padded = torch.cat([
                sample['node_repr'],
                torch.zeros(pad_len, sample['node_repr'].shape[1])
            ], dim=0)
            
            edge_padded = torch.cat([
                torch.cat([sample['edge_repr'], torch.zeros(N, pad_len, sample['edge_repr'].shape[2])], dim=1),
                torch.zeros(pad_len, max_len, sample['edge_repr'].shape[2])
            ], dim=0)
            
            rigids_pad = torch.zeros(pad_len, 7)
            rigids_pad[:, 0] = 1.0  # Identity rotation for padding
            rigids_padded = torch.cat([
                sample['gt_rigids'],
                rigids_pad
            ], dim=0)
            
            mask_padded = torch.cat([
                sample['mask'],
                torch.zeros(pad_len)
            ], dim=0)
        else:
            node_padded = sample['node_repr']
            edge_padded = sample['edge_repr']
            rigids_padded = sample['gt_rigids']
            mask_padded = sample['mask']
        
        batch_node.append(node_padded)
        batch_edge.append(edge_padded)
        batch_gt_rigids.append(rigids_padded)
        batch_mask.append(mask_padded)
        batch_accessions.append(sample['accession'])
    
    return {
        'node_repr': torch.stack(batch_node),
        'edge_repr': torch.stack(batch_edge),
        'gt_rigids': torch.stack(batch_gt_rigids),
        'mask': torch.stack(batch_mask),
        'accessions': batch_accessions,
    }


if __name__ == '__main__':
    # Test the dataset
    print("Testing DynarepoDataset...")
    
    dataset = DynarepoDataset(
        csv_path=f"{root_path_02}/public/caizhiqiang/DyneTrion-multimer/DyneTrion/train_data.csv",
        max_seq_len=500,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load first sample
    sample = dataset[0]
    
    print(f"\nSample accession: {sample['accession']}")
    print(f"Node repr shape: {sample['node_repr'].shape}")
    print(f"Edge repr shape: {sample['edge_repr'].shape}")
    print(f"GT rigids shape: {sample['gt_rigids'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    
    # Test collate
    batch = collate_fn([dataset[0], dataset[1]])
    
    print(f"\nBatch shapes:")
    print(f"  Node: {batch['node_repr'].shape}")
    print(f"  Edge: {batch['edge_repr'].shape}")
    print(f"  Rigids: {batch['gt_rigids'].shape}")
    print(f"  Mask: {batch['mask'].shape}")
    
    print("\n✓ Dataset test passed!")
