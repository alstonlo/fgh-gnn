from rdkit import Chem
from rdkit.Chem.rdchem import BondType

ogb_bond_list = (BondType.SINGLE,
                 BondType.DOUBLE,
                 BondType.TRIPLE,
                 BondType.AROMATIC)


def ogb_graph_to_mol(graph):

    mol = Chem.RWMol()

    # make atoms
    for atom_feat in graph["node_feat"]:

        atomic_num = atom_feat[0].item() + 1
        if not (1 <= atomic_num <= 118):
            raise ValueError("Invalid Atomic Number.")

        mol.AddAtom(Chem.Atom(atomic_num))

    # make bonds
    src, dst = graph["edge_index"]

    for i, j, bond_feat in zip(src, dst, graph["edge_feat"]):

        i, j = i.item(), j.item()
        type_idx = bond_feat[0].item()

        if i >= j:  # prevent doubling of edges
            continue
        if not (0 <= type_idx <= 3):
            raise ValueError("Invalid Bond Type.")

        mol.AddBond(i, j, ogb_bond_list[type_idx])

    mol = mol.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol
