import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch
import torch.nn.functional as F
from dgl.utils import expand_as_pair
from torch import nn


class FGHGNNConvLayer(nn.Module):

    def __init__(self, hidden_dim, gat_num_heads):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = gat_num_heads

        self.bond_gineconv = self.create_gineconv()
        self.overlap_gineconv = self.create_gineconv()
        self.refine_gatconv = self.create_gatconv()
        self.pool_gatconv = self.create_gatconv()

        self.conv = dglnn.HeteroGraphConv({
            'bond': self.bond_gineconv,
            'overlap': self.overlap_gineconv,
            'refine': self.refine_gatconv,
            'pool': self.pool_gatconv,
        })

    def reset_parameters(self):
        self.bond_gineconv.reset_parameters()
        self.overlap_gineconv.reset_parameters()
        self.refine_gatconv.reset_parameters()
        self.pool_gatconv.reset_parameters()

    def forward(self, *args, **kwargs):
        return self.conv(*args, **kwargs)

    def create_gineconv(self):
        apply_func = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.BatchNorm1d(2 * self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )
        return GINEConv(apply_func, 'sum', learn_eps=True)

    def create_gatconv(self):
        return ModifiedGATConv(self.hidden_dim, self.num_heads)


class GINEConv(nn.Module):
    """Adapted from [1] and [2].

    References:
        [1] https://github.com/dmlc/dgl/blob/master/python/dgl/
            nn/pytorch/conv/ginconv.py
        [2] https://github.com/rusty1s/pytorch_geometric/
            blob/master/torch_geometric/nn/conv/gin_conv.py
    """

    def __init__(self,
                 apply_func,
                 aggregator_type,
                 init_eps=0,
                 learn_eps=False):
        super().__init__()
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'
                           .format(aggregator_type))
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

    def reset_parameters(self):
        for module in self.apply_func:
            if isinstance(module, (nn.Linear, nn.BatchNorm1d)):
                module.reset_parameters()

    def forward(self, graph, feat, efeat):

        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)

            graph.srcdata['h'] = feat_src
            graph.edata['w'] = efeat

            graph.update_all(self.message_func, self._reducer('m', 'neigh'))
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            return rst

    @staticmethod
    def message_func(edges):
        return {'m': F.relu(edges.src['h'] + edges.data['w'])}


class ModifiedGATConv(nn.Module):

    def __init__(self, hidden_dim, num_heads):
        super().__init__()

        self.gat_conv = dglnn.GATConv(in_feats=hidden_dim,
                                      out_feats=hidden_dim,
                                      num_heads=num_heads,
                                      feat_drop=0.1,
                                      attn_drop=0.1,
                                      residual=True)

        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(num_heads * hidden_dim),
            nn.ReLU(),
            nn.Linear(num_heads * hidden_dim, hidden_dim),
        )

    def reset_parameters(self):
        self.gat_conv.reset_parameters()
        self.fc_out[1].reset_parameters()
        self.fc_out[3].reset_parameters()

    def forward(self, graph, feat):
        out = self.gat_conv(graph, feat)
        return self.fc_out(out)
