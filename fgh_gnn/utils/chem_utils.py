import itertools

from ogb.utils.features import atom_feature_vector_to_dict
from rdkit import Chem
from rdkit.Chem.rdchem import BondType

ogb_bond_list = (BondType.SINGLE,
                 BondType.DOUBLE,
                 BondType.TRIPLE,
                 BondType.AROMATIC)


def pyg_graph_to_mol(data):
    mol = Chem.RWMol()

    # make atoms
    for atom_feat in data.x:

        feat_dict = atom_feature_vector_to_dict(atom_feat)

        atom = Chem.Atom(feat_dict['atomic_num'])
        atom.SetFormalCharge(feat_dict['formal_charge'])
        atom.SetNumExplicitHs(feat_dict['num_h'])
        atom.SetNumRadicalElectrons(feat_dict['num_rad_e'])
        atom.SetIsAromatic(feat_dict['is_aromatic'])
        mol.AddAtom(atom)

    # make bonds
    src, dst = data.edge_index

    for i, j, bond_feat in zip(src, dst, data.edge_attr):

        i, j = i.item(), j.item()
        type_idx = bond_feat[0].item()

        if i >= j:  # prevent doubling of edges
            continue
        mol.AddBond(i, j, ogb_bond_list[type_idx])

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    Chem.Kekulize(mol)
    return mol


def get_ring_fragments(mol):
    ssr = [set(x) for x in Chem.GetSymmSSSR(mol)]

    # account for bridged compounds
    for ring_a, ring_b in itertools.combinations(ssr, 2):
        if len(ring_a & ring_b) > 2:
            ring_a.update(ring_b)
            ring_b.clear()
    ssr = [r for r in ssr if r]  # clear all empty sets

    return ssr
