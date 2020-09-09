import dgl.function as fn
import torch
import torch.nn.functional as F
from dgl.utils import expand_as_pair
from torch import nn


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
        super(GINEConv, self).__init__()
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
