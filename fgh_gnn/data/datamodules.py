import pytorch_lightning as pl

from .datasets import OGBPropPredDataset


class OGBDataModule(pl.LightningDataModule):

    def __init__(self, name, data_dir, min_freq):
        super(OGBDataModule).__init__()

        self.name = name
        self.data_dir = data_dir
        self.min_freq = min_freq

        # Attributes to be made
        self.dataset = None
        self.split_idx = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        self.dataset = OGBPropPredDataset(self.name, self.data_dir)

    def setup(self, stage=None):

        self.split_idx = self.dataset.get_idx_split()

        test_idx = self.split_idx["test"]

        if (stage == 'fit') or (stage is None):

            ogb_train_set = [self.dataset[i] for i in self.split_idx["train"]]
            ogb_val_set = [self.dataset[i] for i in self.split_idx["valid"]]

            self.train_set = ogb_train_set
            self.val_set = ogb_val_set

        if (stage == 'test') or (stage is None):
            self.test_set = self.dataset[test_idx]
