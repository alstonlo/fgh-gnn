import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from ogb.graphproppred import Evaluator
from pytorch_lightning.core.decorators import auto_move_data
from torch import nn

from .fgh_gnn import FGHGNN


class FGHGNNLightning(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--hidden_channels', type=int, default=150)
        parser.add_argument('--proj_dim', type=int, default=150)
        parser.add_argument('--num_heads', type=int, default=3)
        parser.add_argument('--num_convs', type=int, default=2)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--pdrop', type=float, default=0.1)
        parser.add_argument('--global_pool', type=str, default='mean')
        parser.add_argument('--residual', type=bool, default=True)

        parser.add_argument('--lr', type=float, default=0.00039)

        return parser

    def __init__(self, config):
        super().__init__()

        self.hparams = config
        self.lr = self.hparams.lr

        # variables
        if config.num_classes == 2:  # binary classification
            self.loss_f = nn.BCEWithLogitsLoss()
        else:  # regression
            self.loss_f = nn.MSELoss()
        self.evaluator = Evaluator(config.name)

        self.model = FGHGNN(config)

    @auto_move_data
    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        mask = ~torch.isnan(batch.y)
        y = batch.y
        y_hat = self(batch)
        loss = self.loss_f(y_hat[mask], y[mask])

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss)
        return result

    def validation_step(self, batch, batch_idx):
        result = pl.EvalResult()
        result.y = batch.y
        result.y_hat = self(batch)
        return result

    def validation_epoch_end(self, outputs):
        mask = ~torch.isnan(outputs.y)
        mean_val_loss = self.loss_f(outputs.y_hat[mask], outputs.y[mask])

        eval_metric = self.evaluator.eval({
            'y_pred': outputs.y_hat,
            'y_true': outputs.y
        })[self.evaluator.eval_metric]
        eval_metric = torch.tensor(eval_metric)

        result = pl.EvalResult(checkpoint_on=eval_metric,
                               early_stop_on=eval_metric)
        result.log('metric', eval_metric)
        result.log('val_loss', mean_val_loss)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
