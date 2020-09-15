import optuna
import pathlib
import pytorch_lightning as pl
import torch
from argparse import Namespace
from optuna.integration.pytorch_lightning import (
    PyTorchLightningPruningCallback
)

from fgh_gnn.data import OGBDataModule
from fgh_gnn.models import FGHGNNLightning


class MetricsCallback(pl.Callback):

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def objective(trial):
    data_dir = pathlib.Path(__file__).parents[2] / 'datasets'

    config = {
        'name': 'ogbg-molesol',
        'data_dir': data_dir,
        'num_workers': 0,
        'batch_size': 128,
        'hidden_channels': trial.suggest_int('hidden_channels', 50, 300),
        'proj_dim': trial.suggest_int('proj_dim', 50, 600),
        'num_heads': trial.suggest_int('num_heads', 2, 5),
        'num_convs': trial.suggest_int('num_heads', 2, 5),
        'num_layers': trial.suggest_int('num_layers', 2, 5),
        'pdrop': trial.suggest_float('pdrop', 0.0, 0.5),
        'global_pool': trial.suggest_categorical('global_pool',
                                                 ['mean', 'attention']),
        'residual': trial.suggest_categorical('residual', [True, False]),
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-2),
    }
    config = Namespace(**config)

    # datamodule
    datamodule = OGBDataModule(config)
    datamodule.prepare_data()
    datamodule.setup(stage='fit')

    config.out_dim = datamodule.dataset.num_tasks
    config.num_classes = datamodule.dataset.num_classes

    # make loggers and callbacks
    if datamodule.dataset.eval_metric == 'rmse':
        config.mode = 'min'
    else:
        config.mode = 'max'

    early_stopping = pl.callbacks.EarlyStopping(min_delta=0.0001,
                                                patience=4,
                                                verbose=True,
                                                mode='min')
    prune_callback = PyTorchLightningPruningCallback(
        trial, monitor='val_early_stop_on'
    )
    metrics_callback = MetricsCallback()

    trainer = pl.Trainer(
        logger=False,
        max_epochs=60,
        early_stop_callback=early_stopping,
        checkpoint_callback=False,
        callbacks=[prune_callback, metrics_callback],
        progress_bar_refresh_rate=0,
        gpus=(1 if torch.cuda.is_available() else None),
    )

    model = FGHGNNLightning(config)
    trainer.fit(model, datamodule=datamodule)

    return metrics_callback.metrics[-1]['val_early_stop_on'].item()


def optimize_main():
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)

    study = optuna.create_study(storage="sqlite:///fgh_gnn.db",
                                sampler=sampler,
                                pruner=pruner,
                                study_name='optimize-fgh-gnn',
                                direction='minimize',
                                load_if_exists=True)

    study.optimize(objective,
                   n_trials=100,
                   timeout=86400)

    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    best_trial = study.best_trial

    print(f"  Value: {best_trial.value}")

    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
