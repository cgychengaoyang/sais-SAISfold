# RNA Training Data Preparation for DyneTrion

This document describes how to prepare RNA training data for DyneTrion using mmCIF structures and Protenix embeddings.

## Overview

DyneTrion can now train on RNA structures using the IPA (Invariant Point Attention) module for RNA support. The data preparation pipeline handles:

- Parsing RNA structures from mmCIF files
- Extracting all RNA atoms (not just backbone)
- Creating trajectory data compatible with DyneTrion's PdbDataset
- Generating CSV entries for training
- Producing Protenix-compatible restype encoding (A=22, C=23, G=24, U=25)

## Quick Start

### Prepare Single RNA Structure

```bash
python examples/prepare_rna_training_data.py \
    --pdb_id 2KMJ \
    --output_dir rna_data \
    --n_frames 100
```

This will:
1. Download `2KMJ.cif` from RCSB PDB
2. Parse the RNA structure (28 nucleotides)
3. Create 100-frame trajectory data
4. Generate `rna_data/2KMJ/metadata.csv` for training

### Test Compatibility

```bash
python examples/test_rna_data_compatibility.py \
    --csv_path rna_data/2KMJ/metadata.csv
```

## Data Format

### Output Files

For each PDB ID, the following files are created:

```
rna_data/{pdb_id}/
├── {pdb_id}.cif          # Downloaded mmCIF structure
├── {pdb_id}.npz          # Trajectory data
└── metadata.csv          # Training metadata
```

### NPZ File Structure

The `.npz` file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `aatype` | (N, 21) | One-hot encoding (A=0, C=1, G=2, U=3) |
| `restype_protenix` | (N,) | Protenix encoding (A=22, C=23, G=24, U=25) |
| `is_rna` | (N,) | Boolean flag for RNA residues |
| `all_atom_positions` | (F, N, 37, 3) | Trajectory frames (F frames, 37 atoms) |
| `all_atom_mask` | (N, 37) | Atom presence mask |
| `residue_index` | (N,) | Residue indices |
| `chain_index` | (N,) | Chain indices |
| `sequence` | str | RNA sequence string |

Where:
- N = number of residues
- F = number of trajectory frames
- 37 = atom37 format (compatible with protein atom37)

### CSV Format

The `metadata.csv` file contains:

| Column | Description |
|--------|-------------|
| `pdb_id` | PDB identifier |
| `total_seq_len` | Number of residues |
| `seq_len` | Sequence length list |
| `sequence` | RNA sequence |
| `pos_path` | Path to trajectory NPZ |
| `embed_path` | Path to Protenix embedding |
| `traj_path` | Path to trajectory (same as pos_path) |

## Advanced Usage

### Custom Paths

```bash
python examples/prepare_rna_training_data.py \
    --pdb_id 2KMJ \
    --cif_path /path/to/2KMJ.cif \
    --embed_path /path/to/2KMJ_seed101.npz \
    --output_dir rna_data \
    --n_frames 100
```

### Batch Processing

For multiple RNA structures:

```python
#!/usr/bin/env python3
import subprocess

pdb_ids = ['2KMJ', '1Y26', '2N6R', '5V3F']

for pdb_id in pdb_ids:
    subprocess.run([
        'python', 'examples/prepare_rna_training_data.py',
        '--pdb_id', pdb_id,
        '--output_dir', 'rna_data',
        '--n_frames', '100'
    ])
```

### Merge Multiple CSVs

```python
import pandas as pd
import glob

csv_files = glob.glob('rna_data/*/metadata.csv')
df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
df.to_csv('rna_data/rna_training.csv', index=False)
```

## RNA Atom Naming

The script handles standard RNA atom naming from mmCIF/PDB:

### Sugar Atoms
- C1', C2', C3', C4', C5'
- O2', O3', O4', O5'

### Phosphate Atoms
- P, OP1, OP2, OP3

### Base Atoms (Guanine)
- N1, N2, N3, N7, N9
- C2, C4, C5, C6, C8
- O6

### Base Atoms (Adenine)
- N1, N3, N6, N7, N9
- C2, C4, C5, C6, C8

### Base Atoms (Cytosine)
- N1, N3, N4
- C2, C4, C5, C6
- O2

### Base Atoms (Uracil)
- N1, N3
- C2, C4, C5, C6
- O2, O4

## Protenix Embeddings

The script expects Protenix embeddings at:

```
/inspire/ssd/project/sais-bio/public/xiangwenkai/Protenix_v1/extract_embedding/output/embeddings_npz/{pdb_id}_seed101.npz
```

Embedding format:
- `single_s`: (N, 384) node representations
- `pair_z`: (N, N, 128) edge representations
- `asym_id`: (N,) chain identifiers
- `residue_index`: (N,) residue indices

## Training Configuration

Example training configuration for RNA:

```yaml
# DyneTrion RNA training config
data:
  csv_path: rna_data/rna_training.csv
  val_csv_path: rna_data/rna_training.csv
  motion_number: 2
  ref_number: 1
  frame_time: 16
  frame_sample_step: 1
  filtering:
    train_max_len: 512
    val_max_len: 512
  crop:
    enabled: true
    crop_size: 256

model:
  node_input_embed_size: 384  # Match Protenix single_s dim
  node_embed_size: 256
  edge_embed_size: 128        # Match Protenix pair_z dim
  ipa:
    c_s: 256
    c_z: 128
    c_hidden: 256
    no_heads: 8
```

## Hybrid Protein-RNA Structures

For structures containing both protein and RNA:

1. Prepare protein chains using standard protein pipeline
2. Prepare RNA chains using this RNA pipeline
3. Combine the data with proper chain identifiers

The `is_rna` flag in the NPZ file allows the model to distinguish between protein and RNA residues.

## Troubleshooting

### Embedding Mismatch

If you get embedding shape errors:

```python
# Verify embedding matches structure
import numpy as np
embed = np.load('2KMJ_seed101.npz')
print(f"Embedding residues: {embed['single_s'].shape[0]}")

data = np.load('rna_data/2KMJ/2KMJ.npz')
print(f"Structure residues: {data['aatype'].shape[0]}")
```

### Atom Parsing Issues

To check which atoms are being parsed:

```python
import numpy as np
data = np.load('rna_data/2KMJ/2KMJ.npz')
atom_mask = data['all_atom_mask']
print(f"Atoms per residue: {atom_mask.sum(axis=1)}")
```

### OpenFold Transform Errors

If OpenFold transforms fail:
- Ensure aatype indices are in range 0-20 for OpenFold compatibility
- The data loader handles RNA-to-UNK mapping automatically
- Check that `is_rna` flag is properly set

## Implementation Details

### Restype Encoding

The data uses dual encoding:

1. **Stored encoding** (21-dim one-hot):
   - A=0, C=1, G=2, U=3 (RNA)
   - This is compatible with OpenFold transforms

2. **Protenix encoding** (restype_protenix array):
   - A=22, C=23, G=24, U=25
   - Matches AlphaFold3/Protenix convention

### Data Flow

1. **Parse mmCIF**: Extract RNA coordinates and atom names
2. **Create trajectory**: Generate synthetic frames (or load MD trajectory)
3. **Encode features**: Create aatype, is_rna, restype_protenix arrays
4. **Save NPZ**: Store trajectory data
5. **Create CSV**: Generate metadata for training
6. **DataLoader**: Load and process for model training

### Compatibility with HybridConverter

The RNA data is compatible with the `HybridConverter` class in `src/data/all_atom.py`, which handles:
- RNA-specific all-atom generation using RhoFold converter
- Protein-RNA hybrid structure conversion
- Proper atom masking for mixed structures

## References

- Protenix: https://github.com/bytedance/Protenix
- RhoFold: https://github.com/ml4bio/RhoFold
- OpenFold: https://github.com/aqlaboratory/openfold
- mmCIF format: https://mmcif.wwpdb.org/
