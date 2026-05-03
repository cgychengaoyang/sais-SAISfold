"""Convert PDB/mmCIF files to Protenix-compatible features using the full pipeline."""

import copy
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
from biotite.structure import AtomArray
from Bio.PDB import PDBParser

from src.data.protenix_pipeline.json_to_feature import SampleDictToFeatures
from src.data.protenix_pipeline.data_pipeline import DataPipeline
from src.data.protenix_pipeline.tokenizer import AtomArrayTokenizer
from src.data.protenix_pipeline.featurizer import Featurizer
from src.data.protenix_pipeline.utils import make_dummy_feature, data_type_transform
from src.data.protenix_pipeline.torch_utils import dict_to_tensor


PROTEIN_RESIDUES = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}
RNA_RESIDUES = {'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U'}
DNA_RESIDUES = {'DA': 'A', 'DC': 'C', 'DG': 'G', 'DT': 'T'}


def _load_structure_array(file_path: str):
    """Load a structure file (PDB or mmCIF) into a Biotite AtomArray."""
    path = Path(file_path)
    if path.suffix.lower() in {".cif", ".mmcif"}:
        cif_file = pdbx.CIFFile.read(str(path))
        array = pdbx.get_structure(cif_file, model=1)
    else:
        pdb_file = pdb.PDBFile.read(str(path))
        array = pdb_file.get_structure(model=1)
    # Remove water for sequence extraction (keep in coord overlay)
    array = array[array.res_name != "HOH"]
    return array


def extract_sequence_from_structure(
    structure_array: AtomArray,
    chain_id: Optional[str] = None,
) -> dict:
    """Extract sequences from a structure AtomArray, organized by chain.
    
    Returns:
        dict: {chain_id: {"seq": str, "mol_type": "protein" | "rna" | "dna"}}
    """
    sequences = {}
    chain_ids = sorted(set(structure_array.chain_id))
    
    for cid in chain_ids:
        if chain_id is not None and cid != chain_id:
            continue
        chain_atoms = structure_array[structure_array.chain_id == cid]
        # Get one atom per residue to preserve order
        residues = []
        seen = set()
        for i in range(len(chain_atoms)):
            rid = (int(chain_atoms.res_id[i]), str(chain_atoms.ins_code[i]) if hasattr(chain_atoms, 'ins_code') else "")
            if rid not in seen:
                seen.add(rid)
                residues.append(chain_atoms.res_name[i])
        
        if not residues:
            continue
            
        seq = []
        rna_count = 0
        dna_count = 0
        protein_count = 0
        
        for resname in residues:
            if resname in PROTEIN_RESIDUES:
                seq.append(PROTEIN_RESIDUES[resname])
                protein_count += 1
            elif resname in RNA_RESIDUES:
                seq.append(RNA_RESIDUES[resname])
                rna_count += 1
            elif resname in DNA_RESIDUES:
                seq.append(DNA_RESIDUES[resname])
                dna_count += 1
            else:
                seq.append('X')
        
        if len(seq) == 0:
            continue
            
        # Determine mol_type by majority vote
        counts = {"protein": protein_count, "rna": rna_count, "dna": dna_count}
        mol_type = max(counts, key=counts.get)
        if counts[mol_type] == 0:
            mol_type = "protein"  # default fallback
        
        sequences[cid] = {"seq": "".join(seq), "mol_type": mol_type}
    
    return sequences


def build_protenix_json_from_pdb(
    pdb_path: str,
    chain_id: Optional[str] = None,
) -> dict:
    """Build a Protenix-style input JSON dict from a PDB or mmCIF file.
    
    Args:
        pdb_path: Path to PDB or mmCIF file
        chain_id: If specified, only use this chain. Otherwise use all chains.
        
    Returns:
        Protenix input dict with sequences
    """
    structure_array = _load_structure_array(pdb_path)
    sequences = extract_sequence_from_structure(structure_array, chain_id)
    
    if len(sequences) == 0:
        raise ValueError(f"No valid sequences found in {pdb_path}")
    
    json_dict = {"sequences": []}
    for cid, info in sequences.items():
        mol_type = info["mol_type"]
        seq = info["seq"]
        if mol_type == "rna":
            json_dict["sequences"].append({
                "rnaSequence": {
                    "sequence": seq,
                    "count": 1,
                }
            })
        elif mol_type == "dna":
            json_dict["sequences"].append({
                "dnaSequence": {
                    "sequence": seq,
                    "count": 1,
                }
            })
        else:
            json_dict["sequences"].append({
                "proteinChain": {
                    "sequence": seq,
                    "count": 1,
                }
            })
    
    return json_dict


def overlay_pdb_coordinates(
    atom_array: AtomArray,
    pdb_path: str,
    chain_id: Optional[str] = None,
) -> AtomArray:
    """Replace CCD reference coordinates in atom_array with actual structure coordinates.
    
    Matches by relative residue index and atom_name. Handles chain_id and
    res_id mismatches between JSON-built atom_array and structure file.
    """
    pdb_array = _load_structure_array(pdb_path)
    
    # Filter to target chain if specified
    if chain_id is not None:
        pdb_array = pdb_array[pdb_array.chain_id == chain_id]
    
    # Get unique chains in atom_array (usually just one from JSON)
    atom_array_chains = np.unique(atom_array.chain_id)
    
    # Build lookup from PDB: (relative_res_idx, atom_name) -> coord
    # relative_res_idx is 0-based index within the chain
    pdb_chain_ids = np.unique(pdb_array.chain_id)
    coord_lookups = {}
    for cid in pdb_chain_ids:
        chain_atoms = pdb_array[pdb_array.chain_id == cid]
        res_ids = chain_atoms.res_id
        unique_res_ids = np.unique(res_ids)
        res_id_to_idx = {int(rid): idx for idx, rid in enumerate(unique_res_ids)}
        
        lookup = {}
        for atom in chain_atoms:
            rel_idx = res_id_to_idx[int(atom.res_id)]
            key = (rel_idx, str(atom.atom_name))
            lookup[key] = atom.coord
        coord_lookups[cid] = lookup
    
    # For each chain in atom_array, try to match against PDB chains
    is_resolved = np.zeros(len(atom_array), dtype=bool)
    matched = 0
    
    for atom_chain_idx, atom_cid in enumerate(atom_array_chains):
        chain_mask = atom_array.chain_id == atom_cid
        chain_atoms = atom_array[chain_mask]
        chain_indices = np.where(chain_mask)[0]
        
        # Get relative residue indices for atom_array chain
        unique_res_ids = np.unique(chain_atoms.res_id)
        res_id_to_idx = {int(rid): idx for idx, rid in enumerate(unique_res_ids)}
        
        # Try matching against each PDB chain, pick best match
        best_matched = 0
        best_lookup = None
        
        for pdb_cid, lookup in coord_lookups.items():
            temp_matched = 0
            for i, atom in enumerate(chain_atoms):
                rel_idx = res_id_to_idx[int(atom.res_id)]
                key = (rel_idx, str(atom.atom_name))
                if key in lookup:
                    temp_matched += 1
            if temp_matched > best_matched:
                best_matched = temp_matched
                best_lookup = lookup
                # Update chain_id in atom_array to match PDB
                if atom_cid != pdb_cid:
                    atom_array.chain_id[chain_mask] = pdb_cid
        
        if best_lookup is not None:
            for i, atom in enumerate(chain_atoms):
                rel_idx = res_id_to_idx[int(atom.res_id)]
                key = (rel_idx, str(atom.atom_name))
                if key in best_lookup:
                    atom_array.coord[chain_indices[i]] = best_lookup[key]
                    is_resolved[chain_indices[i]] = True
                    matched += 1
    
    atom_array.set_annotation("is_resolved", is_resolved)
    print(f"Overlayed {matched}/{len(atom_array)} atoms with PDB coordinates")
    return atom_array


def preprocess_pdb_protenix_full(
    pdb_path: str,
    chain_id: Optional[str] = None,
    device: torch.device = torch.device('cpu'),
    use_msa: bool = False,
) -> dict:
    """Full Protenix pipeline preprocessing from PDB.
    
    Args:
        pdb_path: Path to PDB file
        chain_id: Chain to use (None = all chains)
        device: torch device
        use_msa: Whether to use MSA (requires databases)
        
    Returns:
        Dictionary with:
            - input_feature_dict: Protenix input features (with real CCD data)
            - aatype: [N] amino acid type indices
            - atom37_pos: [N, 37, 3] atom positions (from PDB)
            - rigids_0: [N, 7] backbone rigids
            - torsion_angles: from OpenFold transforms on PDB coords
            - And other training-ready features
    """
    # Step 1: Build Protenix JSON from PDB
    json_dict = build_protenix_json_from_pdb(pdb_path, chain_id)
    
    # Step 2: Convert JSON to features using SampleDictToFeatures
    sample2feat = SampleDictToFeatures(json_dict)
    features_dict, atom_array, token_array = sample2feat.get_feature_dict()
    
    # Step 3: Overlay actual PDB coordinates
    atom_array = overlay_pdb_coordinates(atom_array, pdb_path, chain_id)
    
    # Step 4: Get labels from featurizer (uses actual PDB coords now)
    feat = Featurizer(
        cropped_token_array=token_array,
        cropped_atom_array=atom_array,
        ref_pos_augment=False,
        lig_atom_rename=False,
    )
    
    # Update features with featurizer-generated ones
    features_dict.update(feat.get_all_input_features())
    labels_dict = feat.get_labels()
    
    # Step 5: Handle MSA
    entity_to_asym_id = DataPipeline.get_label_entity_id_to_asym_id_int(atom_array)
    
    if use_msa:
        from src.data.protenix_pipeline.msa_featurizer import InferenceMSAFeaturizer
        msa_features = InferenceMSAFeaturizer.make_msa_feature(
            bioassembly=json_dict["sequences"],
            entity_to_asym_id=entity_to_asym_id,
            token_array=token_array,
            atom_array=atom_array,
        )
        if len(msa_features) == 0:
            dummy_feats = ["msa", "template"]
        else:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)
            dummy_feats = ["template"]
    else:
        dummy_feats = ["msa", "template"]
    
    features_dict = make_dummy_feature(features_dict, dummy_feats=dummy_feats)
    features_dict = data_type_transform(features_dict)
    
    # Convert to training-ready format
    N_token = features_dict["token_index"].shape[0]
    N_atom = features_dict["atom_to_token_idx"].shape[0]
    
    # Build dense trunk features (d_lm, v_lm, pad_info)
    from src.model.protenix.model.modules.transformer import rearrange_qk_to_dense_trunk
    with torch.no_grad():
        q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(
            q=[features_dict["ref_pos"], features_dict["ref_space_uid"]],
            k=[features_dict["ref_pos"], features_dict["ref_space_uid"]],
            dim_q=[-2, -1],
            dim_k=[-2, -1],
            n_queries=32,
            n_keys=128,
            compute_mask=True,
        )
        d_lm = q_trunked_list[0][..., None, :] - k_trunked_list[0][..., None, :, :]
        v_lm = (q_trunked_list[1][..., None].int() == k_trunked_list[1][..., None, :].int()).unsqueeze(dim=-1)
        features_dict["d_lm"] = d_lm
        features_dict["v_lm"] = v_lm
        features_dict["pad_info"] = pad_info
    
    # Generate relative position encoding
    from src.model.protenix.model.modules.embedders import RelativePositionEncoding
    relpe = RelativePositionEncoding(r_max=32, s_max=2, c_z=128).to(device)
    relpe.eval()
    with torch.no_grad():
        features_dict = relpe.generate_relp(features_dict)
    
    # Move all tensors in features_dict to target device (recursively)
    def _to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, dict):
            return {k: _to_device(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [_to_device(v) for v in x]
        elif isinstance(x, tuple):
            return tuple(_to_device(v) for v in x)
        return x
    
    for key in list(features_dict.keys()):
        features_dict[key] = _to_device(features_dict[key])
    
    # Extract training features
    coord = labels_dict["coordinate"]  # [N_atom, 3]
    coord_mask = labels_dict["coordinate_mask"]  # [N_atom]
    
    # Build atom37 from token-level data
    # Map atom_array back to per-token atom37
    aatype = features_dict["restype"].argmax(dim=-1).cpu().numpy()
    
    # Build rigids from N, CA, C coordinates
    # Find centre atoms (CA for protein)
    centre_atom_indices = np.where(atom_array.centre_atom_mask == 1)[0]
    n_atoms = []
    ca_atoms = []
    c_atoms = []
    
    for idx in centre_atom_indices:
        # For this token, find N, CA, C atoms
        chain_id = atom_array.chain_id[idx]
        res_id = atom_array.res_id[idx]
        
        n_mask = (atom_array.chain_id == chain_id) & (atom_array.res_id == res_id) & (atom_array.atom_name == "N")
        ca_mask = (atom_array.chain_id == chain_id) & (atom_array.res_id == res_id) & (atom_array.atom_name == "CA")
        c_mask = (atom_array.chain_id == chain_id) & (atom_array.res_id == res_id) & (atom_array.atom_name == "C")
        
        if n_mask.sum() > 0 and ca_mask.sum() > 0 and c_mask.sum() > 0:
            n_atoms.append(atom_array.coord[np.where(n_mask)[0][0]])
            ca_atoms.append(atom_array.coord[np.where(ca_mask)[0][0]])
            c_atoms.append(atom_array.coord[np.where(c_mask)[0][0]])
        else:
            # Fallback: use centre atom for all
            n_atoms.append(atom_array.coord[idx])
            ca_atoms.append(atom_array.coord[idx])
            c_atoms.append(atom_array.coord[idx])
    
    if len(n_atoms) > 0:
        n_pos = torch.tensor(np.stack(n_atoms), dtype=torch.float32, device=device)
        ca_pos = torch.tensor(np.stack(ca_atoms), dtype=torch.float32, device=device)
        c_pos = torch.tensor(np.stack(c_atoms), dtype=torch.float32, device=device)
        
        from openfold.utils import rigid_utils as ru
        bb_rigids = ru.Rigid.from_3_points(n_pos, ca_pos, c_pos)
        rigids_0 = bb_rigids.to_tensor_7()
    else:
        rigids_0 = torch.zeros(len(centre_atom_indices), 7, device=device)
    
    # Compute torsion angles from actual PDB coordinates using OpenFold
    from openfold.data import data_transforms
    from openfold.np import residue_constants as rc
    
    # Build atom37 positions [N_token, 37, 3]
    atom37_pos = torch.zeros(len(centre_atom_indices), 37, 3, device=device)
    atom37_mask = torch.zeros(len(centre_atom_indices), 37, dtype=torch.float32, device=device)
    
    for i, idx in enumerate(centre_atom_indices):
        chain_id_val = atom_array.chain_id[idx]
        res_id_val = atom_array.res_id[idx]
        
        res_mask = (atom_array.chain_id == chain_id_val) & (atom_array.res_id == res_id_val)
        for j, atom in enumerate(atom_array[res_mask]):
            atom_name = atom.atom_name
            if atom_name in rc.atom_order:
                atom_idx = rc.atom_order[atom_name]
                atom37_pos[i, atom_idx] = torch.tensor(atom.coord, dtype=torch.float32, device=device)
                atom37_mask[i, atom_idx] = 1.0
    
    aatype_torch = torch.tensor(aatype, dtype=torch.long, device=device)
    
    prot_feats = {
        'aatype': aatype_torch.unsqueeze(0),
        'all_atom_positions': atom37_pos.unsqueeze(0),
        'all_atom_mask': atom37_mask.unsqueeze(0),
    }
    torsion_angles_feats = data_transforms.atom37_to_torsion_angles()(prot_feats)
    
    torsion_angles = torsion_angles_feats['torsion_angles_sin_cos'][0]
    alt_torsion_angles = torsion_angles_feats['alt_torsion_angles_sin_cos'][0]
    torsion_angles_mask = torsion_angles_feats['torsion_angles_mask'][0]
    
    return {
        'input_feature_dict': features_dict,
        'aatype': aatype_torch,
        'atom37_pos': atom37_pos,
        'atom37_mask': atom37_mask,
        'rigids_0': rigids_0,
        'torsion_angles_sin_cos': torsion_angles,
        'alt_torsion_angles_sin_cos': alt_torsion_angles,
        'torsion_angles_mask': torsion_angles_mask,
        'seq_idx': torch.arange(len(aatype), device=device).unsqueeze(0),
        'residue_index': features_dict['residue_index'].unsqueeze(0) if features_dict['residue_index'].dim() == 1 else features_dict['residue_index'],
        'atom_array': atom_array,
        'token_array': token_array,
    }
