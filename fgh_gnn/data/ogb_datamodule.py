import dgl
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .ogb_dataset import OGBPropPredDataset


class OGBDataModule(pl.LightningDataModule):

    def __init__(self, name, data_dir, min_count, batch_size, num_workers=0):
        super(OGBDataModule).__init__()

        self.name = name
        self.data_dir = data_dir
        self.min_count = min_count
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Attributes to be assigned
        self.dataset = None
        self.split_idx = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        self.dataset = OGBPropPredDataset(name=self.name,
                                          root=self.data_dir,
                                          min_count=self.min_count)

    def setup(self, stage=None):

        self.split_idx = self.dataset.get_idx_split()

        if (stage == 'fit') or (stage is None):
            self.train_set = self.dataset[self.split_idx['train']]
            self.val_set = self.dataset[self.split_idx['valid']]

        if (stage == 'test') or (stage is None):
            self.test_set = self.dataset[self.split_idx['test']]

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=_collate_fn)


def _collate_fn(batch):
    graphs, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.stack(labels)
