import pathlib

import pytorch_lightning as pl
from ogb.graphproppred import GraphPropPredDataset

from fgh_gnn.utils import analyze_func_groups
from fgh_gnn.utils import ogb_graph_to_mol


class OGBDataModule(pl.LightningDataModule):

    def __init__(self, name, root):
        super(OGBDataModule).__init__()

        self.name = name
        self.root = root

        # Attributes to be made
        self.dataset = None
        self.split_idx = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        self.dataset = GraphPropPredDataset(self.name, self.root)
        self.split_idx = self.dataset.get_idx_split()

        # functional group analysis
        train_graphs = [self.dataset[i][0] for i in self.split_idx["train"]]
        train_mols = map(ogb_graph_to_mol, train_graphs)

        fg_stats = analyze_func_groups(train_mols)

        # save analysis and vocab to csv file
        dataset_dir = pathlib.Path(self.dataset.root)
        fg_stats.to_csv(dataset_dir / "functional_groups.csv")

    def setup(self, stage=None):

        train_idx = self.split_idx["train"]
        val_idx = self.split_idx["valid"]
        test_idx = self.split_idx["test"]

        if (stage == 'fit') or (stage is None):
            self.train_set = self.dataset[train_idx]
            self.val_set = self.dataset[val_idx]

        if (stage == 'test') or (stage is None):
            self.test_set = self.dataset[test_idx]
