"""
All functional groups (and their SMARTS) taken from Daylight [1] and [2].

Reference:
    [1] https://www.daylight.com/dayhtml_tutorials/languages/smarts/
        smarts_examples.html
    [2] https://github.com/rdkit/rdkit/blob/master/Data/FunctionalGroups.txt
"""

import json

import pandas as pd
import pathlib
from rdkit import Chem

from .chem_utils import get_ring_fragments


def _load_fgroup_json():
    fgroup_path = pathlib.Path(__file__).parent / "fgroups.json"
    with open(fgroup_path, 'r') as f:
        return json.load(f)


FGROUP_SMARTS = _load_fgroup_json()

FGROUP_MOLS = {name: Chem.MolFromSmarts(s)
               for name, s in FGROUP_SMARTS.items()}


def count_func_groups(mol):
    return {name: len(mol.GetSubstructMatches(fgroup_query))
            for name, fgroup_query in FGROUP_MOLS.items()}


def analyze_fgroups_and_rings(smiles_batch):
    batch_len = 0
    fgroup_count = {name: 0 for name, _ in FGROUP_MOLS.items()}
    ring_count = {}

    for smiles in smiles_batch:

        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        Chem.Kekulize(mol)

        # functional groups
        for name, count in count_func_groups(mol).items():
            fgroup_count[name] += count

        # rings
        for ring_idxs in get_ring_fragments(mol):
            ring_smiles = Chem.MolFragmentToSmiles(mol, list(ring_idxs),
                                                   isomericSmiles=False,
                                                   kekuleSmiles=True)
            ring_count[ring_smiles] = ring_count.get(ring_smiles, 0) + 1

        batch_len += 1

    data = []

    for fgroup_name in sorted(fgroup_count.keys()):
        row = {
            'name': fgroup_name,
            'smarts': FGROUP_SMARTS[fgroup_name],
            'type': 'fgroup',
            'count': fgroup_count[fgroup_name]
        }
        data.append(row)

    for ring_smiles in sorted(ring_count.keys()):
        ring = Chem.MolFromSmiles(ring_smiles)
        row = {
            'name': ring_smiles,
            'smarts': Chem.MolToSmarts(ring, isomericSmiles=False),
            'type': 'ring',
            'count': ring_count[ring_smiles]
        }
        data.append(row)

    df = pd.DataFrame(data=data)
    df['freq'] = df.apply(lambda r: (r['count'] / batch_len), axis=1)
    return df
