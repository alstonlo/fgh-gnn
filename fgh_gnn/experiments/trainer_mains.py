import pathlib
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from argparse import ArgumentParser
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from fgh_gnn.data import OGBDataModule
from fgh_gnn.models import FGHGNNLightning


def train_on_ogb():
    # arguments
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--min_delta', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=10)

    parser = OGBDataModule.add_datamodule_specific_args(parser)
    parser = FGHGNNLightning.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    config = parser.parse_args()

    # seed everything
    pl.trainer.seed_everything(config.seed)

    # make save directories
    project_dir = pathlib.Path(__file__).parents[2]
    logger_dir = project_dir / 'logs'
    ckpt_dir = project_dir / 'results' / config.name / f'v{config.seed}'

    logger_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # datamodule
    datamodule = OGBDataModule(config)
    datamodule.prepare_data()
    datamodule.setup(stage='fit')

    config.vocab_dim = len(datamodule.dataset.vocab)
    config.out_dim = datamodule.dataset.num_tasks
    config.num_classes = datamodule.dataset.num_classes

    # make loggers and callbacks
    if datamodule.dataset.eval_metric == 'rmse':
        config.mode = 'min'
    else:
        config.mode = 'max'

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=logger_dir,
                                             name=config.name,
                                             version=f'v{config.seed}',
                                             log_graph=False)
    early_stopping = EarlyStopping(min_delta=config.min_delta,
                                   patience=config.patience,
                                   verbose=True,
                                   mode=config.mode)
    ckpting = ModelCheckpoint(filepath=(ckpt_dir / '{epoch}-{metric:.4f}'),
                              verbose=True,
                              save_top_k=3,
                              mode=config.mode)

    # training
    trainer = pl.Trainer.from_argparse_args(
        config,
        checkpoint_callback=ckpting,
        early_stop_callback=early_stopping,
        deterministic=True,
        logger=tb_logger,
        progress_bar_refresh_rate=1
    )

    model = FGHGNNLightning(config)

    trainer.fit(model, datamodule=datamodule)
