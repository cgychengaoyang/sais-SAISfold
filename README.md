# DyneTrion-Multimer: SE(3) Diffusion for Protein-RNA Complexes

DyneTrion-Multimer extends the DyneTrion framework to predict and generate structures of **protein-RNA complexes** using an SE(3) diffusion model. It supports both:

- **Full temporal dynamics** (multi-frame trajectory prediction)
- **Single-structure prediction** (seq→structure, including multimer complexes)

This repo contains the full training and inference pipeline, including Protenix-compatible data preprocessing, MSA integration, and RNA-aware backbone construction.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Complete Pipeline](#complete-pipeline)
   - [1. Data Preparation](#1-data-preparation)
   - [2. Preprocess PDB/mmCIF](#2-preprocess-pdbmmtif)
   - [3. Training](#3-training)
   - [4. Evaluation & Sampling](#4-evaluation--sampling)
4. [Directory Structure](#directory-structure)
5. [Troubleshooting](#troubleshooting)

---

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 with CUDA
- Biopython, Biotite, OpenFold utils
- Protenix dependencies (for CCD featurization)
- RhoFold (for RNA all-atom reconstruction)

### Setup

```bash
git clone <repo-url>
cd SAISfold
pip install -e .
# or install dependencies manually:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install biopython biotite omegaconf hydra-core pandas numpy scipy wandb
```

> **Note:** If you plan to use MSA features, you also need `hmmer`, `kalign`, and the sequence databases (see [MSA section](#with-msa-requires-databases)).

---

## Quick Start

Train directly from a PDB file (no pre-computed embeddings required):

```bash
python examples/train_end_to_end_from_pdb.py \
    --pdb_path data/1biv.pdb \
    --output_dir outputs/quickstart \
    --epochs 50 \
    --lr 1e-4
```

For **multimer complexes** with the simpler backbone-only score network:

```bash
python examples/train_9dcf_multimer_style.py
```

This will train on the preprocessed `overfitting_data/9DCF/` and save the best model to `outputs/9dcf_multimer_style/best_model.pth`.

---

## Complete Pipeline

### 1. Data Preparation

#### Option A: With MSA (requires databases)

To use real MSA features, you need:
- `uniref100`
- `mmseqs2` databases
- `uniclust30`
- `pdb_mmseqs` / `distillation_mmseqs`

Set the paths in your environment or config, then run:

```bash
python examples/preprocess_pdb_protenix.py \
    --pdb_path data/your_structure.pdb \
    --output_path features/your_structure.pt \
    --use_msa
```

#### Option B: Without MSA (single-sequence, faster)

For most overfitting / small-scale experiments, MSA is not required. The pipeline will use dummy MSA/template features:

```bash
python examples/preprocess_pdb_protenix.py \
    --pdb_path data/your_structure.pdb \
    --output_path features/your_structure.pt
```

This produces a `.pt` file containing:
- `aatype`: residue type indices
- `rigids_0`: backbone rigid bodies (N-CA-C for protein, C4′-C1′-P for RNA)
- `torsion_angles_sin_cos`, `torsion_angles_mask`
- `input_feature_dict`: Protenix-compatible token/atom features

#### Bulk preprocessing from mmCIF archives

We include the official Protenix preprocessing tools under `scripts/`:

```bash
# Generate bioassembly pickles + CSV indices from a directory of mmCIF files
python scripts/prepare_training_data.py \
    --mmcif_dir /path/to/mmcifs \
    --output_dir /path/to/bioassemblies \
    --num_workers 8

# Generate CCD RDKit cache (needed for some ligands)
python scripts/gen_ccd_cache.py

# Run MSA search for inference JSONs
python scripts/msa_search.py
```

The full Protenix MSA pipeline steps are also available in `scripts/msa/`.

---

### 2. Preprocess PDB/mmCIF

The pipeline works with both **PDB** and **mmCIF** files. For multimer complexes, mmCIF is preferred because it contains `asym_id` and `residue_index` metadata needed to align with Protenix embeddings.

#### Example: preparing a single structure

```bash
python examples/preprocess_pdb_protenix.py \
    --pdb_path data/9dcf.cif \
    --output_path overfitting_data/9DCF/features.pt
```

#### Example: preparing RNA data

```bash
python examples/prepare_rna_training_data.py \
    --pdb_id 2KMJ \
    --output_dir rna_data/2KMJ \
    --n_frames 100
```

For a **full multimer complex** (e.g., 9DCF) that needs token-level alignment with Protenix embeddings, use the dedicated setup script:

```bash
python examples/setup_9dcf_data.py
```

This writes:
- `overfitting_data/9DCF/structure.npz`
- `overfitting_data/9DCF/trajectory.npz`
- `overfitting_data/9DCF/embedding.npz` (copied from precomputed Protenix embeddings)
- `overfitting_data/9DCF/9dcf_data.csv`

---

### 3. Training

We provide **three training modes** depending on your use case.

#### 3A. Full DyneTrion (temporal / dynamics)

Use this for the full model with reference+motion frames, cropping, and auxiliary losses.

```bash
# Create CSVs pointing to your trajectory.npz + embedding.npz pairs
# Then run the main training script via Hydra:

python DyneTrion/train_DyneTrion.py \
    experiment.num_gpus=1 \
    experiment.batch_size=1 \
    data.csv_path=datasets/train_data.csv \
    data.val_csv_path=datasets/val_data.csv \
    data.frame_time=16 \
    data.motion_number=2 \
    data.crop.enabled=True \
    data.crop.crop_size=384 \
    experiment.num_epoch=750 \
    experiment.rot_loss_weight=7.0
```

See `script_train/train_crop384.sh` for a full example.

**CSV format:**
```csv
pdb_id,pos_path,embed_path,seq,total_seq_len,seq_len
9DCF,overfitting_data/9DCF/trajectory.npz,overfitting_data/9DCF/embedding.npz,MET,476,[476]
```

#### 3B. End-to-End from PDB (no precomputed embeddings)

Train a lightweight end-to-end model directly from a PDB file. This uses the Protenix featurizer on-the-fly and does **not** require precomputed embeddings.

```bash
python examples/train_end_to_end_from_pdb.py \
    --pdb_path data/1biv.pdb \
    --output_dir outputs/end_to_end_1biv \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 1 \
    --use_protenix_pipeline
```

For **batched multimer complexes**:

```bash
python examples/train_end_to_end_complex.py \
    --features_dir features/ \
    --output_dir outputs/end_to_end_complex \
    --epochs 50 \
    --batch_size 2
```

#### 3C. Multimer-style Backbone (direct score network)

This is the simplest and most robust option for **single-structure overfitting** on large protein-RNA complexes. It uses `DyneTrionScoreNet` (IPA only, no PairFormer) and converges quickly.

```bash
python examples/train_9dcf_multimer_style.py
```

You can edit the script to point at your own preprocessed data directory.

> **Tip:** On the 9DCF complex (476 residues), this reaches ~3.5 Å CA/C4′ RMSD after 5000 epochs.

---

### 4. Evaluation & Sampling

#### Full DyneTrion inference

```bash
python DyneTrion/inference_DyneTrion.py \
    eval.weights_path=outputs/ckpt/best_model.pth \
    experiment.use_ddp=False \
    experiment.batch_size=1 \
    data.test_csv_path=datasets/test_data.csv
```

You can also use the Protenix inference runner directly:

```bash
python scripts/inference.py \
    --input_json inference.json \
    --output_dir predictions/
```

#### End-to-end inference

The end-to-end scripts evaluate automatically at the end of training. To run sampling from a saved checkpoint:

```bash
python examples/inference_9dcf.py
```

This loads `outputs/overfit_9dcf/final_model.pth`, runs 50-step reverse diffusion, and saves:
- `gt_9dcf.pdb`
- `pred_9dcf_raw.pdb`
- `pred_9dcf_aligned.pdb`
- `sampling_results_9dcf.npz`

#### Multimer-style inference

```bash
python examples/inference_9dcf_multimer_style.py
```

This evaluates direct denoising at multiple noise levels and runs 50-step reverse diffusion. Outputs are saved to:
- `outputs/9dcf_multimer_style/gt.pdb`
- `outputs/9dcf_multimer_style/pred_aligned.pdb`

---

## Directory Structure

```
SAISfold/
├── DyneTrion/                 # Main training & inference entry points
│   ├── train_DyneTrion.py
│   ├── inference_DyneTrion.py
│   └── config/
│       └── train_DyneTrion.yaml
├── src/
│   ├── data/                  # Data loaders, diffusers, preprocessing
│   │   ├── protenix_pipeline/ # Protenix-compatible featurization
│   │   ├── se3_diffuser.py
│   │   ├── all_atom.py
│   │   └── pdb_preprocessor.py
│   ├── model/                 # Model architectures
│   │   ├── diffusion_4d_network_dynamic.py  # Full DyneTrion
│   │   ├── score_based_ipa.py               # Backbone score net
│   │   └── end_to_end_scorenet.py           # End-to-end model
│   └── experiments/
├── examples/                  # Standalone training & inference scripts
│   ├── preprocess_pdb_protenix.py
│   ├── train_end_to_end_from_pdb.py
│   ├── train_end_to_end_complex.py
│   ├── train_9dcf_multimer_style.py
│   ├── inference_9dcf.py
│   ├── inference_9dcf_multimer_style.py
│   └── setup_9dcf_data.py
├── scripts/                   # CCD cache + Protenix tools
│   ├── gen_ccd_rdkit_cache.py
│   ├── gen_ccd_cache.py
│   ├── prepare_training_data.py
│   ├── msa_search.py
│   ├── inference.py
│   └── msa/
├── outputs/                   # Training outputs & checkpoints
└── overfitting_data/          # Preprocessed small datasets
```

---

## Troubleshooting

### `ValueError: not enough values to unpack` during inference

This usually means the batch tensor shapes don't match what the model expects (e.g., `seq_idx` has an extra batch dimension). Make sure you don't squeeze tensors prematurely in your inference script.

### NaN losses during full-model training

- Disable auxiliary losses initially (`torsion_loss_weight=0`, `bb_atom_loss_weight=0`, `dist_mat_loss_weight=0`).
- Use gradient clipping (`clip_grad_norm=1.0`).
- For protein-RNA complexes, the simpler multimer-style model is more stable.

### RNA atoms look wrong in PDB output

Ensure your data preparation places RNA backbone atoms at the correct indices:
- `atom37` index 0 → C4′
- `atom37` index 1 → C1′
- `atom37` index 2 → N9 (purine) or N1 (pyrimidine)
- `atom37` index 19 → P

If these are missing or at wrong indices, `atom37_to_frames` will build degenerate frames and the reconstructed structure will be physically impossible.

### Checkpoint loading fails with `weights_only` error

PyTorch 2.6 changed the default. Pass `weights_only=False` to `torch.load()`:

```python
torch.load(path, map_location='cpu', weights_only=False)
```

---

## Citation

If you use this code, please cite the original DyneTrion work and the Protenix / AlphaFold3 implementations that this pipeline builds upon.
