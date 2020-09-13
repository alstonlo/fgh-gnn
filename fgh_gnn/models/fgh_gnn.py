import torch.nn.functional as F
import torch_geometric as pyg
from torch import nn

from .embedding import BondEmbedding, NodeEmbedding


class FGHGNN(nn.Module):

    def __init__(self, config):
        super().__init__()

        num_layers = config.num_layers
        vocab_dim = config.vocab_dim
        hidden_channels = config.hidden_channels
        proj_dim = 2 * hidden_channels
        out_dim = config.out_dim
        dropout = config.dropout
        graph_pooling = config.graph_pooling
        residual = config.residual

        self.num_layers = num_layers
        self.residual = residual

        self.node_embedding = NodeEmbedding(vocab_dim, hidden_channels)
        self.bond_embedding = BondEmbedding(hidden_channels)
        self.c2cbond_embedding = nn.Embedding(10, hidden_channels)

        self.dropout = nn.Dropout(p=dropout)

        # atom --> cluster
        apply_func = MLP(hidden_channels, proj_dim)
        atom2c_conv = pyg.nn.GINConv(nn=apply_func, train_eps=True)

        # atom <-- cluster
        apply_func = MLP(hidden_channels, proj_dim)
        c2atom_conv = pyg.nn.GINConv(nn=apply_func, train_eps=True)

        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = FGHGNNConv(hidden_channels=hidden_channels,
                              proj_dim=proj_dim,
                              atom2c_conv=atom2c_conv,
                              c2atom_conv=c2atom_conv)
            self.conv_layers.append(conv)

        self.atom_batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])
        self.cluster_batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])

        if graph_pooling == 'mean':
            self.glob_pool_atom = pyg.nn.global_mean_pool
            self.glob_pool_cluster = pyg.nn.global_mean_pool

        elif graph_pooling == 'global_attention':
            gate_nn = nn.Sequential(
                MLP(hidden_channels, proj_dim),
                nn.Linear(proj_dim, 1)
            )
            self.glob_pool_atom = pyg.nn.GlobalAttention(gate_nn)

            gate_nn = nn.Sequential(
                MLP(hidden_channels, proj_dim),
                nn.Linear(proj_dim, 1)
            )
            self.glob_pool_cluster = pyg.nn.GlobalAttention(gate_nn)

        else:
            raise ValueError("Invalid graph_pooling argument.")

        self.classify = nn.Sequential(
            nn.Linear(hidden_channels, 2 * hidden_channels),
            nn.ReLU(),
            nn.Linear(2 * hidden_channels, out_dim),
        )

    def forward(self, data):
        x = self.node_embedding(data.x, x_type='atom')
        x_cl = self.node_embedding(data.x_cluster, x_type='cluster')
        edge_attr = self.bond_embedding(data.edge_attr)
        c2c_edge_attr = self.c2cbond_embedding(data.c2c_edge_attr)

        for layer in range(self.num_layers):
            h, h_cl = self.conv_layers[layer](
                x=x,
                edge_index=data.edge_index,
                edge_attr=edge_attr,
                x_cl=x_cl,
                c2c_edge_index=data.c2c_edge_index,
                c2c_edge_attr=c2c_edge_attr,
                atom2c_edge_index=data.atom2c_edge_index,
                c2atom_edge_index=data.c2atom_edge_index
            )

            h = self.atom_batch_norms[layer](h)
            h_cl = self.cluster_batch_norms[layer](h_cl)

            if layer == self.num_layers - 1:
                h = self.dropout(h)
                h_cl = self.dropout(h_cl)
            else:
                h = self.dropout(F.relu(h))
                h_cl = self.dropout(F.relu(h_cl))

            if self.residual:
                h = x + h
                h_cl = x_cl + h_cl

            x, x_cl = h, h_cl

        atom_pool = self.glob_pool_atom(x, data.x_batch)
        cluster_pool = self.glob_pool_cluster(x_cl, data.x_cluster_batch)

        return self.classify(atom_pool + cluster_pool)


class FGHGNNConv(nn.Module):

    def __init__(self, hidden_channels, proj_dim,
                 atom2c_conv, c2atom_conv):
        super().__init__()

        # atom <--> atom
        apply_func = MLP(hidden_channels, proj_dim)
        self.atom_gineconv = pyg.nn.GINEConv(nn=apply_func, train_eps=True)

        # cluster <--> cluster
        apply_func = MLP(hidden_channels, proj_dim)
        self.cluster_gineconv = pyg.nn.GINEConv(nn=apply_func, train_eps=True)

        # atom <--> cluster
        self.atom2c_conv = atom2c_conv
        self.c2atom_conv = c2atom_conv

        # (atom <--> atom) + (atom <-- cluster)
        self.merge_atom_lin = nn.Linear(proj_dim, hidden_channels)

        # (cluster <--> cluster) + (atom --> cluster)
        self.merge_cluster_lin = nn.Linear(proj_dim, hidden_channels)

    def forward(self, x, edge_index, edge_attr,
                x_cl, c2c_edge_index, c2c_edge_attr,
                atom2c_edge_index, c2atom_edge_index):

        # atom <--> atom
        h = self.atom_gineconv(x, edge_index, edge_attr)

        # cluster <--> cluster
        h_cl = self.cluster_gineconv(x_cl, c2c_edge_index, c2c_edge_attr)

        # atom --> cluster
        h_atom2c = self.atom2c_conv((x, x_cl), atom2c_edge_index,
                                    size=(x.size(0), x_cl.size(0)))

        # atom <-- cluster
        h_c2atom = self.c2atom_conv((x_cl, x), c2atom_edge_index,
                                    size=(x_cl.size(0), x.size(0)))

        h = self.merge_atom_lin(h + h_c2atom)
        h_cl = self.merge_cluster_lin(h_cl + h_atom2c)

        return h, h_cl


class MLP(nn.Module):

    def __init__(self, *dims):
        super().__init__()

        module_list = []
        for dim_prev, dim_curr in zip(dims, dims[1:]):
            module_list.extend([
                nn.Linear(dim_prev, dim_curr),
                nn.BatchNorm1d(dim_curr),
                nn.ReLU()
            ])

        self.mlp = nn.Sequential(*module_list)

    def forward(self, x):
        return self.mlp(x)
