import pandas as pd
from ogb.graphproppred.dataset_pyg import PygGraphPropPredDataset
from rdkit import Chem

import fgh_gnn.utils as utils
from .graph_builder import FGroupHetGraphBuilder


class OGBPropPredDataset(PygGraphPropPredDataset):

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt", \
               'vocab.csv', \
               'fgroup_and_rings.csv'

    def __init__(self, name, root, min_count):

        self.min_count = min_count

        super().__init__(name=name, root=root)

        self.vocab = pd.read_csv(self.processed_paths[1])

    def process(self):
        vocab = self._build_vocab()
        self.pre_transform = FGroupHetGraphBuilder(vocab)

        super().process()

    def _build_vocab(self):

        # analyze functional groups and rings
        split_idx = self.get_idx_split()
        train_idx = set(i.item() for i in split_idx["train"])

        train_smiles = pd.read_csv(
            self.root / "mapping" / "mol.csv.gz",
            skiprows=(lambda x: (x != 0) and (x not in train_idx)),
        )
        train_smiles = train_smiles['smiles'].tolist()

        fgroups_and_rings = utils.analyze_fgroups_and_rings(train_smiles)

        # extract vocab from analysis
        vocab = fgroups_and_rings[fgroups_and_rings['count'] >= self.min_count]

        atom_data = []  # add atom data
        for atomic_num in range(0, 119):
            atom = Chem.Atom(atomic_num)
            atom_data.append({'name': atom.GetSymbol(), 'type': 'atom'})

        vocab = pd.concat([pd.DataFrame(atom_data), vocab], ignore_index=True)
        vocab = vocab.append({'name': 'misc_ring', 'type': 'ring'},
                             ignore_index=True)

        # save for future
        vocab.to_csv(self.processed_paths[1])
        fgroups_and_rings.to_csv(self.processed_paths[2])

        return vocab
