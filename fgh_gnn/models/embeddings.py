import torch
from torch import nn

from fgh_gnn.data import Cluster


class ClusterEncoder(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()

        self.type_embedding = nn.Embedding(len(Cluster.cluster_types),
                                           emb_dim - 1)

    def forward(self, x):
        type_emb = self.type_embedding(x[:, 0])
        size = x[:, 1].unsqueeze(1).float() / Cluster.max_cluster_size

        return torch.cat([type_emb, size], dim=1)


class C2CEdgeEncoder(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()

        self.type_embedding = nn.Embedding(3, emb_dim)

    def forward(self, x):
        return self.type_embedding(x.squeeze(1))
