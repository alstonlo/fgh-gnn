import itertools

from ogb.utils.features import atom_feature_vector_to_dict
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
import copy

ogb_bond_list = (BondType.SINGLE,
                 BondType.DOUBLE,
                 BondType.TRIPLE,
                 BondType.AROMATIC)


def ogb_graph_to_mol(graph):
    mol = Chem.RWMol()

    # make atoms
    for atom_feat in graph["node_feat"]:

        feat_dict = atom_feature_vector_to_dict(atom_feat)

        atom = Chem.Atom(feat_dict['atomic_num'])
        atom.SetFormalCharge(feat_dict['formal_charge'])
        atom.SetNumExplicitHs(feat_dict['num_h'])
        atom.SetNumRadicalElectrons(feat_dict['num_rad_e'])
        atom.SetIsAromatic(feat_dict['is_aromatic'])
        mol.AddAtom(atom)

    # make bonds
    src, dst = graph["edge_index"]

    for i, j, bond_feat in zip(src, dst, graph["edge_feat"]):

        i, j = i.item(), j.item()
        type_idx = bond_feat[0].item()

        if i >= j:  # prevent doubling of edges
            continue
        mol.AddBond(i, j, ogb_bond_list[type_idx])

    mol = mol.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def get_ring_fragments(mol):

    mol = copy.deepcopy(mol)
    Chem.Kekulize(mol)

    ssr = [set(x) for x in Chem.GetSymmSSSR(mol)]

    # account for fused compounds
    for ring_a, ring_b in itertools.combinations(ssr, 2):
        if len(ring_a & ring_b) > 2:
            ring_a.update(ring_b)
            ring_b.clear()
    ssr = [r for r in ssr if r]  # clear all empty sets

    # extract fragments
    rings = set(Chem.MolFragmentToSmiles(mol, list(r),
                                         isomericSmiles=False,
                                         kekuleSmiles=True)
                for r in ssr)
    return rings
