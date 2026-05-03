# Copyright 2021 DeepMind Technologies Limited
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

"""Constants used in AlphaFold."""

import collections
import functools
import os
from typing import List, Mapping, Tuple

import numpy as np
import tree

# Internal import (35fd).


# Distance from one CA to next CA [trans configuration: omega = 180].
ca_ca = 3.80209737096

# Format: The list for each AA type contains chi1, chi2, chi3, chi4 in
# this order (or a relevant subset from chi1 onwards). ALA and GLY don't have
# chi angles so their chi angle lists are empty.
chi_angles_atoms = {
    'ALA': [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    'ARG': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
    'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'CYS': [['N', 'CA', 'CB', 'SG']],
    'GLN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLY': [],
    'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
    'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
    'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'LYS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
    'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'],
            ['CB', 'CG', 'SD', 'CE']],
    'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'PRO': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],
    'SER': [['N', 'CA', 'CB', 'OG']],
    'THR': [['N', 'CA', 'CB', 'OG1']],
    'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'VAL': [['N', 'CA', 'CB', 'CG1']],
}

# If chi angles given in fixed-length array, this matrix determines how to mask
# them for each AA type. The order is as per restype_order (see below).
chi_angles_mask = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [1.0, 1.0, 1.0, 1.0],  # ARG
    [1.0, 1.0, 0.0, 0.0],  # ASN
    [1.0, 1.0, 0.0, 0.0],  # ASP
    [1.0, 0.0, 0.0, 0.0],  # CYS
    [1.0, 1.0, 1.0, 0.0],  # GLN
    [1.0, 1.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [1.0, 1.0, 0.0, 0.0],  # HIS
    [1.0, 1.0, 0.0, 0.0],  # ILE
    [1.0, 1.0, 0.0, 0.0],  # LEU
    [1.0, 1.0, 1.0, 1.0],  # LYS
    [1.0, 1.0, 1.0, 0.0],  # MET
    [1.0, 1.0, 0.0, 0.0],  # PHE
    [1.0, 1.0, 0.0, 0.0],  # PRO
    [1.0, 0.0, 0.0, 0.0],  # SER
    [1.0, 0.0, 0.0, 0.0],  # THR
    [1.0, 1.0, 0.0, 0.0],  # TRP
    [1.0, 1.0, 0.0, 0.0],  # TYR
    [1.0, 0.0, 0.0, 0.0],  # VAL
]

# The following chi angles are pi periodic: they can be rotated by a multiple
# of pi without affecting the structure.
chi_pi_periodic = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [0.0, 0.0, 0.0, 0.0],  # ARG
    [0.0, 0.0, 0.0, 0.0],  # ASN
    [0.0, 1.0, 0.0, 0.0],  # ASP
    [0.0, 0.0, 0.0, 0.0],  # CYS
    [0.0, 0.0, 0.0, 0.0],  # GLN
    [0.0, 0.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [0.0, 0.0, 0.0, 0.0],  # HIS
    [0.0, 0.0, 0.0, 0.0],  # ILE
    [0.0, 0.0, 0.0, 0.0],  # LEU
    [0.0, 0.0, 0.0, 0.0],  # LYS
    [0.0, 0.0, 0.0, 0.0],  # MET
    [0.0, 1.0, 0.0, 0.0],  # PHE
    [0.0, 0.0, 0.0, 0.0],  # PRO
    [0.0, 0.0, 0.0, 0.0],  # SER
    [0.0, 0.0, 0.0, 0.0],  # THR
    [0.0, 0.0, 0.0, 0.0],  # TRP
    [0.0, 1.0, 0.0, 0.0],  # TYR
    [0.0, 0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0, 0.0],  # UNK
]

# Atoms positions relative to the 8 rigid groups, defined by the pre-omega, phi,
# psi and chi angles:
# 0: 'backbone group',
# 1: 'pre-omega-group', (empty)
# 2: 'phi-group', (currently empty, because it defines only hydrogens)
# 3: 'psi-group',
# 4,5,6,7: 'chi1,2,3,4-group'
# The atom positions are relative to the axis-end-atom of the corresponding
# rotation axis. The x-axis is in direction of the rotation axis, and the y-axis
# is defined such that the dihedral-angle-definiting atom (the last entry in
# chi_angles_atoms above) is in the xy-plane (with a positive y-coordinate).
# format: [atomname, group_idx, rel_position]
rigid_group_atom_positions = {
    'ALA': [
        ['N', 0, (-0.525, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.529, -0.774, -1.205)],
        ['O', 3, (0.627, 1.062, 0.000)],
    ],
    'ARG': [
        ['N', 0, (-0.524, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.524, -0.778, -1.209)],
        ['O', 3, (0.626, 1.062, 0.000)],
        ['CG', 4, (0.616, 1.390, -0.000)],
        ['CD', 5, (0.564, 1.414, 0.000)],
        ['NE', 6, (0.539, 1.357, -0.000)],
        ['NH1', 7, (0.206, 2.301, 0.000)],
        ['NH2', 7, (2.078, 0.978, -0.000)],
        ['CZ', 7, (0.758, 1.093, -0.000)],
    ],
    'ASN': [
        ['N', 0, (-0.536, 1.357, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.531, -0.787, -1.200)],
        ['O', 3, (0.625, 1.062, 0.000)],
        ['CG', 4, (0.584, 1.399, 0.000)],
        ['ND2', 5, (0.593, -1.188, 0.001)],
        ['OD1', 5, (0.633, 1.059, 0.000)],
    ],
    'ASP': [
        ['N', 0, (-0.525, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, 0.000, -0.000)],
        ['CB', 0, (-0.526, -0.778, -1.208)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.593, 1.398, -0.000)],
        ['OD1', 5, (0.610, 1.091, 0.000)],
        ['OD2', 5, (0.592, -1.101, -0.003)],
    ],
    'CYS': [
        ['N', 0, (-0.522, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, 0.000, 0.000)],
        ['CB', 0, (-0.519, -0.773, -1.212)],
        ['O', 3, (0.625, 1.062, -0.000)],
        ['SG', 4, (0.728, 1.653, 0.000)],
    ],
    'GLN': [
        ['N', 0, (-0.526, 1.361, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, 0.000)],
        ['CB', 0, (-0.525, -0.779, -1.207)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.615, 1.393, 0.000)],
        ['CD', 5, (0.587, 1.399, -0.000)],
        ['NE2', 6, (0.593, -1.189, -0.001)],
        ['OE1', 6, (0.634, 1.060, 0.000)],
    ],
    'GLU': [
        ['N', 0, (-0.528, 1.361, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.526, -0.781, -1.207)],
        ['O', 3, (0.626, 1.062, 0.000)],
        ['CG', 4, (0.615, 1.392, 0.000)],
        ['CD', 5, (0.600, 1.397, 0.000)],
        ['OE1', 6, (0.607, 1.095, -0.000)],
        ['OE2', 6, (0.589, -1.104, -0.001)],
    ],
    'GLY': [
        ['N', 0, (-0.572, 1.337, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.517, -0.000, -0.000)],
        ['O', 3, (0.626, 1.062, -0.000)],
    ],
    'HIS': [
        ['N', 0, (-0.527, 1.360, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, 0.000, 0.000)],
        ['CB', 0, (-0.525, -0.778, -1.208)],
        ['O', 3, (0.625, 1.063, 0.000)],
        ['CG', 4, (0.600, 1.370, -0.000)],
        ['CD2', 5, (0.889, -1.021, 0.003)],
        ['ND1', 5, (0.744, 1.160, -0.000)],
        ['CE1', 5, (2.030, 0.851, 0.002)],
        ['NE2', 5, (2.145, -0.466, 0.004)],
    ],
    'ILE': [
        ['N', 0, (-0.493, 1.373, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, -0.000)],
        ['CB', 0, (-0.536, -0.793, -1.213)],
        ['O', 3, (0.627, 1.062, -0.000)],
        ['CG1', 4, (0.534, 1.437, -0.000)],
        ['CG2', 4, (0.540, -0.785, -1.199)],
        ['CD1', 5, (0.619, 1.391, 0.000)],
    ],
    'LEU': [
        ['N', 0, (-0.520, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.522, -0.773, -1.214)],
        ['O', 3, (0.625, 1.063, -0.000)],
        ['CG', 4, (0.678, 1.371, 0.000)],
        ['CD1', 5, (0.530, 1.430, -0.000)],
        ['CD2', 5, (0.535, -0.774, 1.200)],
    ],
    'LYS': [
        ['N', 0, (-0.526, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, 0.000)],
        ['CB', 0, (-0.524, -0.778, -1.208)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.619, 1.390, 0.000)],
        ['CD', 5, (0.559, 1.417, 0.000)],
        ['CE', 6, (0.560, 1.416, 0.000)],
        ['NZ', 7, (0.554, 1.387, 0.000)],
    ],
    'MET': [
        ['N', 0, (-0.521, 1.364, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, 0.000, 0.000)],
        ['CB', 0, (-0.523, -0.776, -1.210)],
        ['O', 3, (0.625, 1.062, -0.000)],
        ['CG', 4, (0.613, 1.391, -0.000)],
        ['SD', 5, (0.703, 1.695, 0.000)],
        ['CE', 6, (0.320, 1.786, -0.000)],
    ],
    'PHE': [
        ['N', 0, (-0.518, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, 0.000, -0.000)],
        ['CB', 0, (-0.525, -0.776, -1.212)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.607, 1.377, 0.000)],
        ['CD1', 5, (0.709, 1.195, -0.000)],
        ['CD2', 5, (0.706, -1.196, 0.000)],
        ['CE1', 5, (2.102, 1.198, -0.000)],
        ['CE2', 5, (2.098, -1.201, -0.000)],
        ['CZ', 5, (2.794, -0.003, -0.001)],
    ],
    'PRO': [
        ['N', 0, (-0.566, 1.351, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, 0.000)],
        ['CB', 0, (-0.546, -0.611, -1.293)],
        ['O', 3, (0.621, 1.066, 0.000)],
        ['CG', 4, (0.382, 1.445, 0.0)],
        # ['CD', 5, (0.427, 1.440, 0.0)],
        ['CD', 5, (0.477, 1.424, 0.0)],  # manually made angle 2 degrees larger
    ],
    'SER': [
        ['N', 0, (-0.529, 1.360, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.518, -0.777, -1.211)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['OG', 4, (0.503, 1.325, 0.000)],
    ],
    'THR': [
        ['N', 0, (-0.517, 1.364, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, -0.000)],
        ['CB', 0, (-0.516, -0.793, -1.215)],
        ['O', 3, (0.626, 1.062, 0.000)],
        ['CG2', 4, (0.550, -0.718, -1.228)],
        ['OG1', 4, (0.472, 1.353, 0.000)],
    ],
    'TRP': [
        ['N', 0, (-0.521, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, 0.000)],
        ['CB', 0, (-0.523, -0.776, -1.212)],
        ['O', 3, (0.627, 1.062, 0.000)],
        ['CG', 4, (0.609, 1.370, -0.000)],
        ['CD1', 5, (0.824, 1.091, 0.000)],
        ['CD2', 5, (0.854, -1.148, -0.005)],
        ['CE2', 5, (2.186, -0.678, -0.007)],
        ['CE3', 5, (0.622, -2.530, -0.007)],
        ['NE1', 5, (2.140, 0.690, -0.004)],
        ['CH2', 5, (3.028, -2.890, -0.013)],
        ['CZ2', 5, (3.283, -1.543, -0.011)],
        ['CZ3', 5, (1.715, -3.389, -0.011)],
    ],
    'TYR': [
        ['N', 0, (-0.522, 1.362, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, -0.000, -0.000)],
        ['CB', 0, (-0.522, -0.776, -1.213)],
        ['O', 3, (0.627, 1.062, -0.000)],
        ['CG', 4, (0.607, 1.382, -0.000)],
        ['CD1', 5, (0.716, 1.195, -0.000)],
        ['CD2', 5, (0.713, -1.194, -0.001)],
        ['CE1', 5, (2.107, 1.200, -0.002)],
        ['CE2', 5, (2.104, -1.201, -0.003)],
        ['OH', 5, (4.168, -0.002, -0.005)],
        ['CZ', 5, (2.791, -0.001, -0.003)],
    ],
    'VAL': [
        ['N', 0, (-0.494, 1.373, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, -0.000)],
        ['CB', 0, (-0.533, -0.795, -1.213)],
        ['O', 3, (0.627, 1.062, -0.000)],
        ['CG1', 4, (0.540, 1.429, -0.000)],
        ['CG2', 4, (0.533, -0.776, 1.203)],
    ],
}

# A list of atoms (excluding hydrogen) for each AA type. PDB naming convention.
residue_atoms = {
    'ALA': ['C', 'CA', 'CB', 'N', 'O'],
    'ARG': ['C', 'CA', 'CB', 'CG', 'CD', 'CZ', 'N', 'NE', 'O', 'NH1', 'NH2'],
    'ASP': ['C', 'CA', 'CB', 'CG', 'N', 'O', 'OD1', 'OD2'],
    'ASN': ['C', 'CA', 'CB', 'CG', 'N', 'ND2', 'O', 'OD1'],
    'CYS': ['C', 'CA', 'CB', 'N', 'O', 'SG'],
    'GLU': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O', 'OE1', 'OE2'],
    'GLN': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'NE2', 'O', 'OE1'],
    'GLY': ['C', 'CA', 'N', 'O'],
    'HIS': ['C', 'CA', 'CB', 'CG', 'CD2', 'CE1', 'N', 'ND1', 'NE2', 'O'],
    'ILE': ['C', 'CA', 'CB', 'CG1', 'CG2', 'CD1', 'N', 'O'],
    'LEU': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'N', 'O'],
    'LYS': ['C', 'CA', 'CB', 'CG', 'CD', 'CE', 'N', 'NZ', 'O'],
    'MET': ['C', 'CA', 'CB', 'CG', 'CE', 'N', 'O', 'SD'],
    'PHE': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O'],
    'PRO': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O'],
    'SER': ['C', 'CA', 'CB', 'N', 'O', 'OG'],
    'THR': ['C', 'CA', 'CB', 'CG2', 'N', 'O', 'OG1'],
    'TRP': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'CZ2', 'CZ3',
            'CH2', 'N', 'NE1', 'O'],
    'TYR': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O',
            'OH'],
    'VAL': ['C', 'CA', 'CB', 'CG1', 'CG2', 'N', 'O']
}

# Naming swaps for ambiguous atom names.
# Due to symmetries in the amino acids the naming of atoms is ambiguous in
# 4 of the 20 amino acids.
# (The LDDT paper lists 7 amino acids as ambiguous, but the naming ambiguities
# in LEU, VAL and ARG can be resolved by using the 3d constellations of
# the 'ambiguous' atoms and their neighbours)
residue_atom_renaming_swaps = {
    'ASP': {'OD1': 'OD2'},
    'GLU': {'OE1': 'OE2'},
    'PHE': {'CD1': 'CD2', 'CE1': 'CE2'},
    'TYR': {'CD1': 'CD2', 'CE1': 'CE2'},
}

# Van der Waals radii [Angstroem] of the atoms (from Wikipedia)
van_der_waals_radius = {
    'C': 1.7,
    'N': 1.55,
    'O': 1.52,
    'S': 1.8,
}

Bond = collections.namedtuple(
    'Bond', ['atom1_name', 'atom2_name', 'length', 'stddev'])
BondAngle = collections.namedtuple(
    'BondAngle',
    ['atom1_name', 'atom2_name', 'atom3name', 'angle_rad', 'stddev'])


@functools.lru_cache(maxsize=None)
def load_stereo_chemical_props() -> Tuple[Mapping[str, List[Bond]],
                                          Mapping[str, List[Bond]],
                                          Mapping[str, List[BondAngle]]]:
  """Load stereo_chemical_props.txt into a nice structure.

  Load literature values for bond lengths and bond angles and translate
  bond angles into the length of the opposite edge of the triangle
  ("residue_virtual_bonds").

  Returns:
    residue_bonds: Dict that maps resname -> list of Bond tuples.
    residue_virtual_bonds: Dict that maps resname -> list of Bond tuples.
    residue_bond_angles: Dict that maps resname -> list of BondAngle tuples.
  """
  stereo_chemical_props_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), 'stereo_chemical_props.txt'
  )
  with open(stereo_chemical_props_path, 'rt') as f:
    stereo_chemical_props = f.read()
  lines_iter = iter(stereo_chemical_props.splitlines())
  # Load bond lengths.
  residue_bonds = {}
  next(lines_iter)  # Skip header line.
  for line in lines_iter:
    if line.strip() == '-':
      break
    bond, resname, length, stddev = line.split()
    atom1, atom2 = bond.split('-')
    if resname not in residue_bonds:
      residue_bonds[resname] = []
    residue_bonds[resname].append(
        Bond(atom1, atom2, float(length), float(stddev)))
  residue_bonds['UNK'] = []

  # Load bond angles.
  residue_bond_angles = {}
  next(lines_iter)  # Skip empty line.
  next(lines_iter)  # Skip header line.
  for line in lines_iter:
    if line.strip() == '-':
      break
    bond, resname, angle_degree, stddev_degree = line.split()
    atom1, atom2, atom3 = bond.split('-')
    if resname not in residue_bond_angles:
      residue_bond_angles[resname] = []
    residue_bond_angles[resname].append(
        BondAngle(atom1, atom2, atom3,
                  float(angle_degree) / 180. * np.pi,
                  float(stddev_degree) / 180. * np.pi))
  residue_bond_angles['UNK'] = []

  def make_bond_key(atom1_name, atom2_name):
    """Unique key to lookup bonds."""
    return '-'.join(sorted([atom1_name, atom2_name]))

  # Translate bond angles into distances ("virtual bonds").
  residue_virtual_bonds = {}
  for resname, bond_angles in residue_bond_angles.items():
    # Create a fast lookup dict for bond lengths.
    bond_cache = {}
    for b in residue_bonds[resname]:
      bond_cache[make_bond_key(b.atom1_name, b.atom2_name)] = b
    residue_virtual_bonds[resname] = []
    for ba in bond_angles:
      bond1 = bond_cache[make_bond_key(ba.atom1_name, ba.atom2_name)]
      bond2 = bond_cache[make_bond_key(ba.atom2_name, ba.atom3name)]

      # Compute distance between atom1 and atom3 using the law of cosines
      # c^2 = a^2 + b^2 - 2ab*cos(gamma).
      gamma = ba.angle_rad
      length = np.sqrt(bond1.length**2 + bond2.length**2
                       - 2 * bond1.length * bond2.length * np.cos(gamma))

      # Propagation of uncertainty assuming uncorrelated errors.
      dl_outer = 0.5 / length
      dl_dgamma = (2 * bond1.length * bond2.length * np.sin(gamma)) * dl_outer
      dl_db1 = (2 * bond1.length - 2 * bond2.length * np.cos(gamma)) * dl_outer
      dl_db2 = (2 * bond2.length - 2 * bond1.length * np.cos(gamma)) * dl_outer
      stddev = np.sqrt((dl_dgamma * ba.stddev)**2 +
                       (dl_db1 * bond1.stddev)**2 +
                       (dl_db2 * bond2.stddev)**2)
      residue_virtual_bonds[resname].append(
          Bond(ba.atom1_name, ba.atom3name, length, stddev))

  return (residue_bonds,
          residue_virtual_bonds,
          residue_bond_angles)


# Between-residue bond lengths for general bonds (first element) and for Proline
# (second element).
between_res_bond_length_c_n = [1.329, 1.341]
between_res_bond_length_stddev_c_n = [0.014, 0.016]

# Between-residue cos_angles.
between_res_cos_angles_c_n_ca = [-0.5203, 0.0353]  # degrees: 121.352 +- 2.315
between_res_cos_angles_ca_c_n = [-0.4473, 0.0311]  # degrees: 116.568 +- 1.995

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37.

# A compact atom encoding with 14 columns
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
restype_name_to_atom14_names = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],
    'GLY': ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'UNK': ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],

}
# pylint: enable=line-too-long
# pylint: enable=bad-whitespace


# This is the standard residue order when coding AA type as a number.
# Reproduce it by taking 3-letter AA codes and sorting them alphabetically.
restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.
unk_restype_index = restype_num  # Catch-all index for unknown restypes.

restypes_with_x = restypes + ['X']
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}


def sequence_to_onehot(
    sequence: str,
    mapping: Mapping[str, int],
    map_unknown_to_x: bool = False) -> np.ndarray:
  """Maps the given sequence into a one-hot encoded matrix.

  Args:
    sequence: An amino acid sequence.
    mapping: A dictionary mapping amino acids to integers.
    map_unknown_to_x: If True, any amino acid that is not in the mapping will be
      mapped to the unknown amino acid 'X'. If the mapping doesn't contain
      amino acid 'X', an error will be thrown. If False, any amino acid not in
      the mapping will throw an error.

  Returns:
    A numpy array of shape (seq_len, num_unique_aas) with one-hot encoding of
    the sequence.

  Raises:
    ValueError: If the mapping doesn't contain values from 0 to
      num_unique_aas - 1 without any gaps.
  """
  num_entries = max(mapping.values()) + 1

  if sorted(set(mapping.values())) != list(range(num_entries)):
    raise ValueError('The mapping must have values from 0 to num_unique_aas-1 '
                     'without any gaps. Got: %s' % sorted(mapping.values()))

  one_hot_arr = np.zeros((len(sequence), num_entries), dtype=int)

  for aa_index, aa_type in enumerate(sequence):
    if map_unknown_to_x:
      if aa_type.isalpha() and aa_type.isupper():
        aa_id = mapping.get(aa_type, mapping['X'])
      else:
        raise ValueError(f'Invalid character in the sequence: {aa_type}')
    else:
      aa_id = mapping[aa_type]
    one_hot_arr[aa_index, aa_id] = 1

  return one_hot_arr


restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}


# NB: restype_3to1 differs from Bio.PDB.protein_letters_3to1 by being a simple
# 1-to-1 mapping of 3 letter names to one letter names. The latter contains
# many more, and less common, three letter names as keys and maps many of these
# to the same one letter name (including 'X' and 'U' which we don't use here).
restype_3to1 = {v: k for k, v in restype_1to3.items()}

# Define a restype name for all unknown residues.
unk_restype = 'UNK'

resnames = [restype_1to3[r] for r in restypes] + [unk_restype]
resname_to_idx = {resname: i for i, resname in enumerate(resnames)}


# The mapping here uses hhblits convention, so that B is mapped to D, J and O
# are mapped to X, U is mapped to C, and Z is mapped to E. Other than that the
# remaining 20 amino acids are kept in alphabetical order.
# There are 2 non-amino acid codes, X (representing any amino acid) and
# "-" representing a missing amino acid in an alignment.  The id for these
# codes is put at the end (20 and 21) so that they can easily be ignored if
# desired.
HHBLITS_AA_TO_ID = {
    'A': 0,
    'B': 2,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'J': 20,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'O': 20,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'U': 1,
    'V': 17,
    'W': 18,
    'X': 20,
    'Y': 19,
    'Z': 3,
    '-': 21,
}

# Partial inversion of HHBLITS_AA_TO_ID.
ID_TO_HHBLITS_AA = {
    0: 'A',
    1: 'C',  # Also U.
    2: 'D',  # Also B.
    3: 'E',  # Also Z.
    4: 'F',
    5: 'G',
    6: 'H',
    7: 'I',
    8: 'K',
    9: 'L',
    10: 'M',
    11: 'N',
    12: 'P',
    13: 'Q',
    14: 'R',
    15: 'S',
    16: 'T',
    17: 'V',
    18: 'W',
    19: 'Y',
    20: 'X',  # Includes J and O.
    21: '-',
}

restypes_with_x_and_gap = restypes + ['X', '-']
MAP_HHBLITS_AATYPE_TO_OUR_AATYPE = tuple(
    restypes_with_x_and_gap.index(ID_TO_HHBLITS_AA[i])
    for i in range(len(restypes_with_x_and_gap)))


def _make_standard_atom_mask() -> np.ndarray:
  """Returns [num_res_types, num_atom_types] mask array."""
  # +1 to account for unknown (all 0s).
  mask = np.zeros([restype_num + 1, atom_type_num], dtype=int)
  for restype, restype_letter in enumerate(restypes):
    restype_name = restype_1to3[restype_letter]
    atom_names = residue_atoms[restype_name]
    for atom_name in atom_names:
      atom_type = atom_order[atom_name]
      mask[restype, atom_type] = 1
  return mask


STANDARD_ATOM_MASK = _make_standard_atom_mask()


# A one hot representation for the first and second atoms defining the axis
# of rotation for each chi-angle in each residue.
def chi_angle_atom(atom_index: int) -> np.ndarray:
  """Define chi-angle rigid groups via one-hot representations."""
  chi_angles_index = {}
  one_hots = []

  for k, v in chi_angles_atoms.items():
    indices = [atom_types.index(s[atom_index]) for s in v]
    indices.extend([-1]*(4-len(indices)))
    chi_angles_index[k] = indices

  for r in restypes:
    res3 = restype_1to3[r]
    one_hot = np.eye(atom_type_num)[chi_angles_index[res3]]
    one_hots.append(one_hot)

  one_hots.append(np.zeros([4, atom_type_num]))  # Add zeros for residue `X`.
  one_hot = np.stack(one_hots, axis=0)
  one_hot = np.transpose(one_hot, [0, 2, 1])

  return one_hot

chi_atom_1_one_hot = chi_angle_atom(1)
chi_atom_2_one_hot = chi_angle_atom(2)

# An array like chi_angles_atoms but using indices rather than names.
chi_angles_atom_indices = [chi_angles_atoms[restype_1to3[r]] for r in restypes]
chi_angles_atom_indices = tree.map_structure(
    lambda atom_name: atom_order[atom_name], chi_angles_atom_indices)
chi_angles_atom_indices = np.array([
    chi_atoms + ([[0, 0, 0, 0]] * (4 - len(chi_atoms)))
    for chi_atoms in chi_angles_atom_indices])

# Mapping from (res_name, atom_name) pairs to the atom's chi group index
# and atom index within that group.
chi_groups_for_atom = collections.defaultdict(list)
for res_name, chi_angle_atoms_for_res in chi_angles_atoms.items():
  for chi_group_i, chi_group in enumerate(chi_angle_atoms_for_res):
    for atom_i, atom in enumerate(chi_group):
      chi_groups_for_atom[(res_name, atom)].append((chi_group_i, atom_i))
chi_groups_for_atom = dict(chi_groups_for_atom)


def _make_rigid_transformation_4x4(ex, ey, translation):
  """Create a rigid 4x4 transformation matrix from two axes and transl."""
  # Normalize ex.
  ex_normalized = ex / np.linalg.norm(ex)

  # make ey perpendicular to ex
  ey_normalized = ey - np.dot(ey, ex_normalized) * ex_normalized
  ey_normalized /= np.linalg.norm(ey_normalized)

  # compute ez as cross product
  eznorm = np.cross(ex_normalized, ey_normalized)
  m = np.stack([ex_normalized, ey_normalized, eznorm, translation]).transpose()
  m = np.concatenate([m, [[0., 0., 0., 1.]]], axis=0)
  return m


# create an array with (restype, atomtype) --> rigid_group_idx
# and an array with (restype, atomtype, coord) for the atom positions
# and compute affine transformation matrices (4,4) from one rigid group to the
# previous group
restype_atom37_to_rigid_group = np.zeros([21, 37], dtype=int)
restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
restype_atom37_rigid_group_positions = np.zeros([21, 37, 3], dtype=np.float32)
restype_atom14_to_rigid_group = np.zeros([21, 14], dtype=int)
restype_atom14_mask = np.zeros([21, 14], dtype=np.float32)
restype_atom14_rigid_group_positions = np.zeros([21, 14, 3], dtype=np.float32)
restype_rigid_group_default_frame = np.zeros([21, 8, 4, 4], dtype=np.float32)


def _make_rigid_group_constants():
  """Fill the arrays above."""
  for restype, restype_letter in enumerate(restypes):
    resname = restype_1to3[restype_letter]
    for atomname, group_idx, atom_position in rigid_group_atom_positions[
        resname]:
      atomtype = atom_order[atomname]
      restype_atom37_to_rigid_group[restype, atomtype] = group_idx
      restype_atom37_mask[restype, atomtype] = 1
      restype_atom37_rigid_group_positions[restype, atomtype, :] = atom_position

      atom14idx = restype_name_to_atom14_names[resname].index(atomname)
      restype_atom14_to_rigid_group[restype, atom14idx] = group_idx
      restype_atom14_mask[restype, atom14idx] = 1
      restype_atom14_rigid_group_positions[restype,
                                           atom14idx, :] = atom_position

  for restype, restype_letter in enumerate(restypes):
    resname = restype_1to3[restype_letter]
    atom_positions = {name: np.array(pos) for name, _, pos
                      in rigid_group_atom_positions[resname]}

    # backbone to backbone is the identity transform
    restype_rigid_group_default_frame[restype, 0, :, :] = np.eye(4)

    # pre-omega-frame to backbone (currently dummy identity matrix)
    restype_rigid_group_default_frame[restype, 1, :, :] = np.eye(4)

    # phi-frame to backbone
    mat = _make_rigid_transformation_4x4(
        ex=atom_positions['N'] - atom_positions['CA'],
        ey=np.array([1., 0., 0.]),
        translation=atom_positions['N'])
    restype_rigid_group_default_frame[restype, 2, :, :] = mat

    # psi-frame to backbone
    mat = _make_rigid_transformation_4x4(
        ex=atom_positions['C'] - atom_positions['CA'],
        ey=atom_positions['CA'] - atom_positions['N'],
        translation=atom_positions['C'])
    restype_rigid_group_default_frame[restype, 3, :, :] = mat

    # chi1-frame to backbone
    if chi_angles_mask[restype][0]:
      base_atom_names = chi_angles_atoms[resname][0]
      base_atom_positions = [atom_positions[name] for name in base_atom_names]
      mat = _make_rigid_transformation_4x4(
          ex=base_atom_positions[2] - base_atom_positions[1],
          ey=base_atom_positions[0] - base_atom_positions[1],
          translation=base_atom_positions[2])
      restype_rigid_group_default_frame[restype, 4, :, :] = mat

    # chi2-frame to chi1-frame
    # chi3-frame to chi2-frame
    # chi4-frame to chi3-frame
    # luckily all rotation axes for the next frame start at (0,0,0) of the
    # previous frame
    for chi_idx in range(1, 4):
      if chi_angles_mask[restype][chi_idx]:
        axis_end_atom_name = chi_angles_atoms[resname][chi_idx][2]
        axis_end_atom_position = atom_positions[axis_end_atom_name]
        mat = _make_rigid_transformation_4x4(
            ex=axis_end_atom_position,
            ey=np.array([-1., 0., 0.]),
            translation=axis_end_atom_position)
        restype_rigid_group_default_frame[restype, 4 + chi_idx, :, :] = mat


_make_rigid_group_constants()


def make_atom14_dists_bounds(overlap_tolerance=1.5,
                             bond_length_tolerance_factor=15):
  """compute upper and lower bounds for bonds to assess violations."""
  restype_atom14_bond_lower_bound = np.zeros([21, 14, 14], np.float32)
  restype_atom14_bond_upper_bound = np.zeros([21, 14, 14], np.float32)
  restype_atom14_bond_stddev = np.zeros([21, 14, 14], np.float32)
  residue_bonds, residue_virtual_bonds, _ = load_stereo_chemical_props()
  for restype, restype_letter in enumerate(restypes):
    resname = restype_1to3[restype_letter]
    atom_list = restype_name_to_atom14_names[resname]

    # create lower and upper bounds for clashes
    for atom1_idx, atom1_name in enumerate(atom_list):
      if not atom1_name:
        continue
      atom1_radius = van_der_waals_radius[atom1_name[0]]
      for atom2_idx, atom2_name in enumerate(atom_list):
        if (not atom2_name) or atom1_idx == atom2_idx:
          continue
        atom2_radius = van_der_waals_radius[atom2_name[0]]
        lower = atom1_radius + atom2_radius - overlap_tolerance
        upper = 1e10
        restype_atom14_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
        restype_atom14_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
        restype_atom14_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
        restype_atom14_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper

    # overwrite lower and upper bounds for bonds and angles
    for b in residue_bonds[resname] + residue_virtual_bonds[resname]:
      atom1_idx = atom_list.index(b.atom1_name)
      atom2_idx = atom_list.index(b.atom2_name)
      lower = b.length - bond_length_tolerance_factor * b.stddev
      upper = b.length + bond_length_tolerance_factor * b.stddev
      restype_atom14_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
      restype_atom14_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
      restype_atom14_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
      restype_atom14_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper
      restype_atom14_bond_stddev[restype, atom1_idx, atom2_idx] = b.stddev
      restype_atom14_bond_stddev[restype, atom2_idx, atom1_idx] = b.stddev
  return {'lower_bound': restype_atom14_bond_lower_bound,  # shape (21,14,14)
          'upper_bound': restype_atom14_bond_upper_bound,  # shape (21,14,14)
          'stddev': restype_atom14_bond_stddev,  # shape (21,14,14)
         }# ============================================================================
# RNA CONSTANTS - ADDED FOR RNA IPA SUPPORT
# ============================================================================
# Using Protenix/AlphaFold3 encoding: A=21, G=22, C=23, U=24, N=25
# Note: This matches the convention used by HybridConverter and prepare_rna_training_data.py

# RNA residue types (order: A, G, C, U to match Protenix 21-24)
rna_restypes = ['A', 'G', 'C', 'U']
rna_restype_order = {restype: i for i, restype in enumerate(rna_restypes)}
rna_restype_num = len(rna_restypes)  # := 4

# Combined restypes (protein + RNA)
# Protein: 0-19 (20 amino acids), UNK_PROT=20
# RNA: 21-25 (A=21, G=22, C=23, U=24, N=25)
hybrid_restypes = restypes + ['UNK_PROT'] + rna_restypes + ['N']  # 26 types (0-25)
hybrid_restype_order = {restype: i for i, restype in enumerate(hybrid_restypes)}
hybrid_restype_num = len(hybrid_restypes)  # := 26 (0-25)

# RNA restype encoding (matching Protenix convention used in this project)
# Protein: 0-19 (20 AA), UNK_PROT=20
# RNA: A=21, G=22, C=23, U=24, N=25
RNA_A = 21
RNA_G = 22
RNA_C = 23
RNA_U = 24
RNA_N = 25

# RNA restype encoding offset (for hybrid_restypes array indexing)
RNA_RESTYPE_OFFSET = 21  # RNA starts at index 21 in hybrid_restypes

# Mapping from nucleotide to index (A=21, G=22, C=23, U=24)
RNA_NUCLEOTIDE_TO_IDX = {
    'A': RNA_A,
    'G': RNA_G,
    'C': RNA_C,
    'U': RNA_U,
}

# RNA atom names per residue (from RhoFold)
rna_residue_atoms = {
    'A': ["C4'", "C1'", 'N9', "C2'", "C3'", "C5'", "O2'", "O3'", "O4'", 
          'N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N6', 'N7', 'C8', "O5'", 'P', 'OP1', 'OP2'],
    'G': ["C4'", "C1'", 'N9', "C2'", "C3'", "C5'", "O2'", "O3'", "O4'",
          'N1', 'N2', 'N3', 'C2', 'C4', 'C5', 'C6', 'N7', 'C8', 'O6', "O5'", 'P', 'OP1', 'OP2'],
    'U': ["C4'", "C1'", 'N1', "C2'", "C3'", "C5'", "O2'", "O3'", "O4'",
          'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6', "O5'", 'P', 'OP1', 'OP2'],
    'C': ["C4'", "C1'", 'N1', "C2'", "C3'", "C5'", "O2'", "O3'", "O4'",
          'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6', "O5'", 'P', 'OP1', 'OP2'],
}

# RNA residue name mappings
rna_restype_1to3 = {'A': 'ADE', 'G': 'GUA', 'U': 'URA', 'C': 'CYT'}
rna_restype_3to1 = {v: k for k, v in rna_restype_1to3.items()}

# Maximum atoms per RNA residue (for atom23 representation)
RNA_ATOM_NUM_MAX = 23

# RNA torsion angle information (4 angles per residue from RhoFold)
rna_torsion_angles = {
    'A': [
        ["C4'", "C1'", 'N9', "C4"],   # angl_0: base rotation
        ['N9',  "C1'", "C4'", "C5'"], # angl_1: sugar
        ["C1'", "C4'", "C5'", "O5'"], # angl_2: backbone
        ["C4'", "C5'", "O5'", "P"],   # angl_3: phosphate connection
    ],
    'G': [
        ["C4'", "C1'", 'N9', "C4"],
        ['N9',  "C1'", "C4'", "C5'"],
        ["C1'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
    ],
    'U': [
        ["C4'", "C1'", 'N1', "C2"],
        ['N1',  "C1'", "C4'", "C5'"],
        ["C1'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
    ],
    'C': [
        ["C4'", "C1'", 'N1', "C2"],
        ['N1',  "C1'", "C4'", "C5'"],
        ["C1'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
    ],
}

# RNA atom ordering for unified representation (23 atoms)
RNA_ATOM_ORDER = {
    "C4'": 0, "C1'": 1, 'N9': 2, 'N1': 2,  # N9 for A/G, N1 for U/C
    "C2'": 3, "C3'": 4, "C5'": 5,
    "O2'": 6, "O3'": 7, "O4'": 8,
    # Base atoms
    'N1_base': 9, 'N2': 10, 'N3': 11, 'N4': 12, 'N6': 13, 'N7': 14, 'N9_base': 15,
    'C2': 16, 'C4': 17, 'C5': 18, 'C6': 19, 'C8': 20,
    'O2': 21, 'O4': 22, 'O6': 23,
    # Backbone
    "O5'": 24, 'P': 25, 'OP1': 26, 'OP2': 27,
}


def is_rna_residue(aatype: int) -> bool:
    """Check if residue type index is RNA (A, G, C, U).
    
    Args:
        aatype: Residue type index (0-25 where 21-24 are RNA: A=21, G=22, C=23, U=24)
        
    Returns:
        True if RNA residue, False otherwise
    """
    return RNA_RESTYPE_OFFSET <= aatype < RNA_RESTYPE_OFFSET + rna_restype_num


def is_protein_residue(aatype: int) -> bool:
    """Check if residue type index is protein (standard amino acids).
    
    Args:
        aatype: Residue type index
        
    Returns:
        True if protein residue (0-19), False otherwise
    """
    return 0 <= aatype < restype_num


def is_gap_or_unk(aatype: int) -> bool:
    """Check if residue type index is GAP or UNK.
    
    Args:
        aatype: Residue type index
        
    Returns:
        True if UNK_PROT (20), False otherwise
        Note: GAP is not used in this project's convention (21 is RNA_A).
    """
    return aatype == 20


def get_residue_type_category(aatype: int) -> str:
    """Get category ('protein', 'rna', 'gap', 'unknown') for residue type.
    
    Args:
        aatype: Residue type index
        
    Returns:
        Category string
    """
    if is_protein_residue(aatype):
        return 'protein'
    elif is_rna_residue(aatype):
        return 'rna'
    elif is_gap_or_unk(aatype):
        return 'gap'
    return 'unknown'


def get_residue_name_from_index(aatype: int) -> str:
    """Get residue name from index.
    
    Args:
        aatype: Residue type index (0-25)
        
    Returns:
        Residue name (e.g., 'ALA', 'GLY', 'A', 'C', 'G', 'U', 'UNK', 'GAP')
    """
    if aatype < restype_num:
        # Protein: 0-19
        return restype_1to3[restypes[aatype]]
    elif aatype == 20:
        # Unknown protein
        return 'UNK_PROT'
    elif is_rna_residue(aatype):
        # RNA: 21-24 (A=21, G=22, C=23, U=24)
        return rna_restypes[aatype - RNA_RESTYPE_OFFSET]
    return 'UNK'


# ============================================================================
# PROTENIX-COMPATIBLE RNA CONSTANTS
# ============================================================================
# AlphaFold3/Protenix restype encoding matches our DyneTrion implementation
# Both use convention for RNA: A=21, G=22, C=23, U=24, N=25

# Protenix restype encoding (AlphaFold3 standard)
# 0-19: Standard amino acids (same as ours)
# 20: Unknown amino acid (protein)
# 21: Gap/mask
# 22-25: RNA (A=22, C=23, G=24, U=25) - ALPHABETICAL ORDER
# 26-29: DNA (A=26, C=27, G=28, T=29)
# 30+: Ligands and other

PROTENIX_RNA_RESTYPES = ['A', 'C', 'G', 'U']  # ALPHABETICAL - same as ours!
PROTENIX_RNA_A = 22
PROTENIX_RNA_C = 23
PROTENIX_RNA_G = 24
PROTENIX_RNA_U = 25

PROTENIX_RNA_RESTYPE_ORDER = {
    'A': 0, 'C': 1, 'G': 2, 'U': 3  # Indices within RNA types
}

# Mapping from DyneTrion internal encoding to Protenix encoding
# Both use the same encoding: A=22, C=23, G=24, U=25 (alphabetical)
# This is now an identity mapping for RNA indices
DYNETRION_TO_PROTENIX_RNA = {
    22: 22,  # A -> A
    23: 23,  # C -> C
    24: 24,  # G -> G
    25: 25,  # U -> U
}

# Reverse mapping
PROTENIX_TO_DYNETRION_RNA = {v: k for k, v in DYNETRION_TO_PROTENIX_RNA.items()}

# Protenix total restype dimension (one-hot)
PROTENIX_RESTYPE_DIM = 32  # 20 AA + UNK + GAP + 4 RNA + 4 DNA + 2 ligand/special

# Restype conversion table (size 256 for efficient lookup)
def _create_restype_conversion_table():
    """Create conversion tables between DyneTrion and Protenix encodings."""
    to_protenix = list(range(256))  # Default: identity mapping
    to_dynetrion = list(range(256))
    
    # Protein: 0-19 same in both
    for i in range(20):
        to_protenix[i] = i
        to_dynetrion[i] = i
    
    # UNK_PROT: 20 -> 20 (protein unknown)
    to_protenix[20] = 20
    to_dynetrion[20] = 20
    
    # GAP: 21 -> 21 (gap/mask)
    to_protenix[21] = 21
    to_dynetrion[21] = 21
    
    # RNA conversions (now identity mapping: 22-25 same in both)
    for dyn_idx, pro_idx in DYNETRION_TO_PROTENIX_RNA.items():
        to_protenix[dyn_idx] = pro_idx
        to_dynetrion[pro_idx] = dyn_idx
    
    return to_protenix, to_dynetrion

DYNETRION_TO_PROTENIX_TABLE, PROTENIX_TO_DYNETRION_TABLE = _create_restype_conversion_table()


def convert_to_protenix_aatype(dynetrion_aatype):
    """Convert DyneTrion aatype to Protenix aatype encoding.
    
    Args:
        dynetrion_aatype: int or array of DyneTrion residue type indices (0-23)
        
    Returns:
        Protenix residue type indices (0-33)
    """
    if isinstance(dynetrion_aatype, int):
        return DYNETRION_TO_PROTENIX_TABLE[dynetrion_aatype]
    elif hasattr(dynetrion_aatype, '__iter__'):
        return [DYNETRION_TO_PROTENIX_TABLE[int(x)] for x in dynetrion_aatype]
    else:
        return DYNETRION_TO_PROTENIX_TABLE[int(dynetrion_aatype)]


def convert_from_protenix_aatype(protenix_aatype):
    """Convert Protenix aatype to DyneTrion aatype encoding.
    
    Args:
        protenix_aatype: int or array of Protenix residue type indices (0-33)
        
    Returns:
        DyneTrion residue type indices (0-23)
    """
    if isinstance(protenix_aatype, int):
        return PROTENIX_TO_DYNETRION_TABLE[protenix_aatype]
    elif hasattr(protenix_aatype, '__iter__'):
        return [PROTENIX_TO_DYNETRION_TABLE[int(x)] for x in protenix_aatype]
    else:
        return PROTENIX_TO_DYNETRION_TABLE[int(protenix_aatype)]


def is_rna_residue_protenix(protenix_aatype: int) -> bool:
    """Check if Protenix residue type is RNA.
    
    Args:
        protenix_aatype: Protenix residue type index (0-31)
        
    Returns:
        True if RNA residue (indices 22-25)
    """
    return 22 <= protenix_aatype <= 25


def create_protenix_biomolecule_flags(aatype):
    """Create Protenix biomolecule type flags from DyneTrion aatype.
    
    Args:
        aatype: DyneTrion residue type indices (0-25)
        
    Returns:
        Dictionary with is_protein, is_rna, is_dna, is_ligand
        Each value is a numpy array of shape (N,) with float values 0.0 or 1.0
    """
    import numpy as np
    
    protenix_aatype = np.array(convert_to_protenix_aatype(aatype))
    
    # Protenix/AlphaFold3 encoding:
    # 0-19: Standard amino acids
    # 20: Unknown amino acid (protein)
    # 21: Gap/mask
    # 22-25: RNA (A=22, C=23, G=24, U=25)
    # 26-29: DNA
    # 30+: Ligands
    is_protein = ((protenix_aatype >= 0) & (protenix_aatype <= 20)).astype(float)
    is_rna = ((protenix_aatype >= 22) & (protenix_aatype <= 25)).astype(float)
    is_dna = ((protenix_aatype >= 26) & (protenix_aatype <= 29)).astype(float)
    is_ligand = (protenix_aatype >= 30).astype(float)
    
    return {
        'is_protein': is_protein,
        'is_rna': is_rna,
        'is_dna': is_dna,
        'is_ligand': is_ligand,
    }


# ============================================================================
# PROTENIX INPUT FEATURE SIZES
# ============================================================================

PROTENIX_FEATURE_DIMS = {
    'restype': 32,           # One-hot: 20 AA + UNK + GAP + 4 RNA + 4 DNA + ligands
    'profile': 32,           # MSA profile
    'deletion_mean': 1,      # MSA deletion values
    'ref_pos': 3,            # Reference positions (x, y, z)
    'ref_charge': 1,         # Atom charges
    'ref_mask': 1,           # Atom mask
    'ref_element': 128,      # One-hot element type
    'ref_atom_name_chars': 256,  # 4 chars x 64 ASCII
}

PROTENIX_CONCATENATED_FEATURE_DIM = 65  # 32 + 32 + 1 (restype + profile + deletion_mean)


# ============================================================================
# RNA ATOM37 EXTENSION - LOAD PRECOMPUTED ARRAYS FROM RHOFFOLD
# ============================================================================
# OpenFold's native atom37 tables only have 21 rows (0-19 proteins + 20 UNK).
# We extend them to 26 rows (0-19 proteins, 20 UNK, 21-25 RNA: A, G, C, U, N)
# so that compute_backbone_atom37() produces actual RNA atoms instead of zeros.

import os as _os

_DATA_DIR = _os.path.dirname(_os.path.abspath(__file__))

try:
    _rna_atom37_to_rigid_group = np.load(_os.path.join(_DATA_DIR, 'rna_atom37_to_rigid_group.npy'))
    _rna_atom37_mask = np.load(_os.path.join(_DATA_DIR, 'rna_atom37_mask.npy'))
    _rna_atom37_positions = np.load(_os.path.join(_DATA_DIR, 'rna_atom37_positions.npy'))
    _rna_default_frames = np.load(_os.path.join(_DATA_DIR, 'rna_default_frames.npy'))
    _rna_atom14_to_rigid_group = np.load(_os.path.join(_DATA_DIR, 'rna_atom14_to_rigid_group.npy'))
    _rna_atom14_mask = np.load(_os.path.join(_DATA_DIR, 'rna_atom14_mask.npy'))
    _rna_atom14_positions = np.load(_os.path.join(_DATA_DIR, 'rna_atom14_positions.npy'))

    restype_atom37_to_rigid_group = _rna_atom37_to_rigid_group
    restype_atom37_mask = _rna_atom37_mask
    restype_atom37_rigid_group_positions = _rna_atom37_positions
    restype_rigid_group_default_frame = _rna_default_frames
    restype_atom14_to_rigid_group = _rna_atom14_to_rigid_group
    restype_atom14_mask = _rna_atom14_mask
    restype_atom14_rigid_group_positions = _rna_atom14_positions
except Exception as _e:
    import warnings
    warnings.warn(f"Failed to load RNA atom37 extensions: {_e}")
