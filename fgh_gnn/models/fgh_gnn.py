import torch
import torch_geometric as pyg
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch import nn

from .embeddings import C2CEdgeEncoder, ClusterEncoder


class FGHGNN(nn.Module):

    def __init__(self, config):
        super().__init__()

        num_layers = config.num_layers
        hidden_channels = config.hidden_channels
        proj_dim = config.proj_dim
        num_convs = config.num_convs
        num_heads = config.num_heads
        pdrop = config.pdrop
        residual = config.residual
        global_pool = config.global_pool
        out_dim = config.out_dim

        self.num_layers = num_layers

        self.atom_encoder = AtomEncoder(hidden_channels)
        self.cluster_encoder = ClusterEncoder(hidden_channels)
        self.bond_encoder = BondEncoder(hidden_channels)
        self.c2c_edge_encoder = C2CEdgeEncoder(hidden_channels)

        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = FGHGNNLayer(hidden_channels=hidden_channels,
                               proj_dim=proj_dim,
                               num_convs=num_convs,
                               num_heads=num_heads,
                               pdrop=pdrop,
                               residual=residual)
            self.conv_layers.append(conv)

        if global_pool == 'mean':
            self.glob_pool_atom = pyg.nn.global_mean_pool
            self.glob_pool_cluster = pyg.nn.global_mean_pool

        elif global_pool == 'attention':
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
            raise ValueError("Invalid global_pool argument.")

        self.classify = nn.Sequential(
            nn.Linear(2 * hidden_channels, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, out_dim),
        )

    def forward(self, data):
        x = self.atom_encoder(data.x)
        x_cl = self.cluster_encoder(data.x_cluster)
        edge_attr = self.bond_encoder(data.edge_attr)
        c2c_edge_attr = self.c2c_edge_encoder(data.c2c_edge_attr)

        for layer in range(self.num_layers):
            x, x_cl = self.conv_layers[layer](
                x=x,
                edge_index=data.edge_index,
                edge_attr=edge_attr,
                x_cl=x_cl,
                c2c_edge_index=data.c2c_edge_index,
                c2c_edge_attr=c2c_edge_attr,
                atom2c_edge_index=data.atom2c_edge_index,
                c2atom_edge_index=data.c2atom_edge_index
            )

        atom_pool = self.glob_pool_atom(x, data.x_batch)
        cluster_pool = self.glob_pool_cluster(x_cl, data.x_cluster_batch)

        out = torch.cat([atom_pool, cluster_pool], dim=1)
        return self.classify(out)


class FGHGNNLayer(nn.Module):

    def __init__(self, hidden_channels, proj_dim, num_heads, num_convs,
                 pdrop, residual, inter_passing=True):
        super().__init__()

        self.num_convs = num_convs
        self.dropout = nn.Dropout(p=pdrop)
        self.residual = residual
        self.inter_passing = inter_passing

        # atom <--> atom
        self.atom_convs = nn.ModuleList()
        for _ in range(num_convs):
            apply_func = MLP(hidden_channels, proj_dim, hidden_channels)
            self.atom_convs.append(
                pyg.nn.GINEConv(nn=apply_func, train_eps=True)
            )

        # cluster <--> cluster
        self.cluster_convs = nn.ModuleList()
        for _ in range(num_convs):
            apply_func = MLP(hidden_channels, proj_dim, hidden_channels)
            self.atom_convs.append(
                pyg.nn.GINEConv(nn=apply_func, train_eps=True)
            )

        # atom --> cluster
        self.pool_conv = pyg.nn.GATConv(
            in_channels=(hidden_channels, hidden_channels),
            out_channels=hidden_channels,
            heads=num_heads,
            dropout=0.1,
            add_self_loops=False
        )
        self.atom2c_mlp = MLP(num_heads * hidden_channels,
                              proj_dim, hidden_channels)

        # atom <-- cluster
        self.unpool_conv = pyg.nn.GATConv(
            in_channels=(hidden_channels, hidden_channels),
            out_channels=hidden_channels,
            heads=num_heads,
            dropout=0.1,
            add_self_loops=False
        )
        self.c2atom_mlp = MLP(num_heads * hidden_channels,
                              proj_dim, hidden_channels)

    def forward(self, x, edge_index, edge_attr,
                x_cl, c2c_edge_index, c2c_edge_attr,
                atom2c_edge_index, c2atom_edge_index):

        # atom <--> atom
        for conv in self.atom_convs:
            h = conv(x, edge_index, edge_attr)
            if self.residual:
                h += x
            x = self.dropout(h)

        # cluster <--> cluster
        for conv in self.cluster_convs:
            h_cl = conv(x_cl, c2c_edge_index, c2c_edge_attr)
            if self.residual:
                h_cl += x_cl
            x_cl = self.dropout(h_cl)

        if self.inter_passing:

            # atom <-- cluster
            h = self.unpool_conv((x_cl, x), c2atom_edge_index,
                                 size=(x_cl.size(0), x.size(0)))
            h = self.c2atom_mlp(h)

            # atom --> cluster
            h_cl = self.pool_conv((x, x_cl), atom2c_edge_index,
                                  size=(x.size(0), x_cl.size(0)))
            h_cl = self.atom2c_mlp(h_cl)

            if self.residual:
                h += x
                h_cl += x_cl

            x = self.dropout(h)
            x_cl = self.dropout(h_cl)

        return x, x_cl


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
