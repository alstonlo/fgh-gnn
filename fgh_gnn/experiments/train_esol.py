import pathlib

from fgh_gnn.data import OGBDataModule


def train():
    root = pathlib.Path(__file__).parents[2] / 'datasets'
    datamodule = OGBDataModule(name='ogbg-molesol',
                               data_dir=root,
                               min_freq=10)
    datamodule.prepare_data()


train()
