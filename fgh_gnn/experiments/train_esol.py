import pathlib

from fgh_gnn.data import OGBDataModule


def train():
    root = pathlib.Path(__file__).parents[2] / 'datasets'
    datamodule = OGBDataModule(name='ogbg-molesol', root=root)
    datamodule.prepare_data()


train()
