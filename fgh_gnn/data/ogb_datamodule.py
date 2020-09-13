import pathlib
import pytorch_lightning as pl
import torch_geometric as tg
from argparse import ArgumentParser

from .ogb_dataset import OGBPropPredDataset


class OGBDataModule(pl.LightningDataModule):

    @staticmethod
    def add_datamodule_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        data_dir = pathlib.Path(__file__).parents[2] / 'datasets'

        parser.add_argument('--name', type=str, default='ogbg-molesol')
        parser.add_argument('--min_count', type=int, default=10)

        parser.add_argument('--data_dir', type=str, default=data_dir)
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--batch_size', type=int, default=64)

        return parser

    def __init__(self, config):
        super().__init__()

        self.name = config.name
        self.min_count = config.min_count
        self.data_dir = config.data_dir
        self.num_workers = config.num_workers
        self.batch_size = config.batch_size

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
        return tg.data.DataLoader(self.train_set,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  follow_batch=['x', 'x_cluster'])

    def val_dataloader(self):
        return tg.data.DataLoader(self.val_set,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  follow_batch=['x', 'x_cluster'])

    def test_dataloader(self):
        return tg.data.DataLoader(self.test_set,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  follow_batch=['x', 'x_cluster'])
