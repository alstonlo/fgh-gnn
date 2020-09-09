from ogb.utils.features import get_atom_feature_dims
from torch import nn


class NodeEmbedding(nn.Module):
    """Adapted from [1].

    References:
        [1] https://github.com/snap-stanford/ogb/blob/master/ogb/
            graphproppred/mol_encoder.py
    """

    def __init__(self, vocab_dim, embd_dim):
        super().__init__()

        vocab_embd = nn.Embedding(vocab_dim, embd_dim)

        # only three cluster types (functional group, ring, atom)
        cluster_type_embd = nn.Embedding(3, embd_dim)

        # atom embedding
        self.atom_embd_list = nn.ModuleList([vocab_embd])

        # skip atom_feature[0], which embeds atomic number
        atom_feature_dims = get_atom_feature_dims()
        for dim in atom_feature_dims[1:]:
            embd = nn.Embedding(dim, embd_dim)
            self.atom_embd_list.append(embd)

        # cluster embedding
        self.cluster_embd_list = nn.ModuleList([vocab_embd, cluster_type_embd])

    def forward(self, x, x_type):

        if x_type == 'atom':
            embd_list = self.atom_embd_list
        elif x_type == 'cluster':
            embd_list = self.cluster_embd_list
        else:
            raise ValueError("Invalid x_type.")

        x_embd = 0
        for i in range(x.shape[1]):
            x_embd += embd_list[i](x[:, i])
        return x_embd
