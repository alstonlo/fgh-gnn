import dgl
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from ogb.graphproppred import Evaluator
from pytorch_lightning.core.decorators import auto_move_data
from torch import nn

from .conv_layers import FGHGNNConvLayer
from .embedding import BondEmbedding, NodeEmbedding


class FGHGNN(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--hidden_dim', type=int, required=True)
        parser.add_argument('--num_conv_layers', type=int, required=True)
        parser.add_argument('--gat_num_heads', type=int, required=True)
        parser.add_argument('--dropout', type=int, required=True)

        parser.add_argument('--lr', type=float, required=True)

        return parser

    def __init__(self, config, dataset):
        super().__init__()

        self.hparams = config
        self.lr = self.hparams.lr

        # variables
        config.vocab_dim = len(dataset.vocab)
        config.out_dim = dataset.num_tasks

        if dataset.num_classes == 2:  # binary classification
            self.loss_f = nn.BCEWithLogitsLoss()
        else:  # regression
            self.loss_f = nn.MSELoss()
        self.evaluator = Evaluator(dataset.name)

        # create nn.Modules
        self.node_embedding = NodeEmbedding(config.vocab_dim,
                                            config.hidden_dim)
        self.bond_embedding = BondEmbedding(config.hidden_dim)
        self.overlap_embedding = nn.Embedding(10, config.hidden_dim)

        self.dropout = nn.Dropout(p=config.dropout)

        self.conv_layers = nn.ModuleList()
        for _ in range(config.num_conv_layers):
            conv = FGHGNNConvLayer(hidden_dim=config.hidden_dim,
                                   gat_num_heads=config.gat_num_heads)
            self.conv_layers.append(conv)

        self.atom_batch_norms = nn.ModuleList([
            nn.BatchNorm1d(config.hidden_dim)
            for _ in range(config.num_conv_layers)
        ])
        self.cluster_batch_norms = nn.ModuleList([
            nn.BatchNorm1d(config.hidden_dim)
            for _ in range(config.num_conv_layers)
        ])

        self.classify = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.out_dim)
        )

    def reset_parameters(self):
        self.node_embedding.reset_parameters()
        self.bond_embedding.reset_parameters()
        self.overlap_embedding.reset_parameters()

        for conv in self.conv_layers:
            conv.reset_parameters()

        for batch_norm in self.atom_batch_norms:
            batch_norm.reset_parameters()
        for batch_norm in self.cluster_batch_norms:
            batch_norm.reset_parameters()

        for module in self.classify:
            if isinstance(module, (nn.Linear, nn.BatchNorm1d)):
                module.reset_parameters()

    @auto_move_data
    def forward(self, g):
        atom_feats = g.nodes['atom'].data['x']
        cluster_feats = g.nodes['cluster'].data['x']
        bond_feats = g.edges['bond'].data['x']
        overlap_feats = g.edges['overlap'].data['x']

        # embed features above
        atom_feats = self.node_embedding(atom_feats, x_type='atom')
        cluster_feats = self.node_embedding(cluster_feats, x_type='cluster')
        bond_feats = self.bond_embedding(bond_feats)
        overlap_feats = self.overlap_embedding(overlap_feats)

        h_0 = {'atom': atom_feats, 'cluster': cluster_feats}

        h_i = h_0
        for conv, atom_bn, cluster_bn in zip(self.conv_layers,
                                             self.atom_batch_norms,
                                             self.cluster_batch_norms):

            mod_args = {'bond': [bond_feats], 'overlap': [overlap_feats]}
            h_i = conv(g, (h_i, h_i), mod_args=mod_args)

            h_i['atom'] = self.dropout(F.relu(atom_bn(h_i['atom'])))
            h_i['cluster'] = self.dropout(F.relu(cluster_bn(h_i['cluster'])))

        with g.local_scope():
            g.ndata['h'] = h_i

            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
            out = self.classify(hg)

            return out

    def training_step(self, batch, batch_idx):
        g, label = batch
        mask = ~torch.isnan(label)

        y = label[mask]
        y_hat = self(g)[mask]
        loss = self.loss_f(y_hat, y)

        result = pl.TrainResult(minimize=loss)
        return result

    def validation_step(self, batch, batch_idx):
        g, label = batch

        result = pl.EvalResult()
        result.y = label
        result.y_hat = self(g)
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
