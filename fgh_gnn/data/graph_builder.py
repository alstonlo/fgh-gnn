import itertools

import dgl
import torch
from rdkit import Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from fgh_gnn.utils import FGROUP_MOLS, get_ring_fragments, ogb_graph_to_mol


class FGroupHetGraphBuilder:

    def __init__(self, vocab):
        self.vocab = vocab

        self.fgroup_vocab = vocab.loc[vocab['type'] == 'fgroup']

        self.ring_vocab = vocab.loc[vocab['type'] == 'ring']
        self.ring_smiles_set = set(self.ring_vocab['name'].unique())
        self.misc_ring_idx = len(vocab)

    def build_fgroup_heterograph(self, raw_graph):

        atom_feats = torch.from_numpy(raw_graph['node_feat'])
        bond_feats = torch.from_numpy(raw_graph['edge_feat'])
        a2a_edges = torch.from_numpy(raw_graph['edge_index'])

        # build tree
        mol = ogb_graph_to_mol(raw_graph)
        clusters = self._make_clusters(mol)
        cluster_feats = torch.tensor([c.features for c in clusters],
                                     dtype=torch.long)

        c2atom_edges, atom2c_edges = self._make_inter_edges(clusters)
        c2c_edges, overlap_feats = \
            self._make_intracluster_edges(raw_graph, clusters)


        data_dict = {
            ('atom', 'bond', 'atom'): (a2a_edges[0], a2a_edges[1]),
            ('cluster', 'refine', 'atom'): (c2atom_edges[0], c2atom_edges[1]),
            ('atom', 'pool', 'cluster'): (atom2c_edges[0], atom2c_edges[1]),
            ('cluster', 'overlap', 'cluster'): (c2c_edges[0], c2c_edges[1])
        }
        num_nodes_dict = {
            'atom': raw_graph['num_nodes'],
            'cluster': len(clusters)
        }

        g = dgl.heterograph(data_dict=data_dict, num_nodes_dict=num_nodes_dict)

        g.nodes['atom'].data['x'] = atom_feats
        g.nodes['cluster'].data['x'] = cluster_feats

        g.edges['bond'].data['x'] = bond_feats
        g.edges['overlap'].data['x'] = overlap_feats

        return g

    def _make_clusters(self, mol):

        clusters = []

        # add all functional groups
        for row in self.fgroup_vocab.itertuples():

            row_idx = row.Index

            fgroup_query = FGROUP_MOLS[row.name]
            matches = mol.GetSubstructMatches(fgroup_query)

            for match_idxs in matches:
                clusters.append(Cluster(row_idx, 'fgroup', match_idxs))

        # add all rings
        for ring_idxs in get_ring_fragments(mol):

            ring_smiles = Chem.MolFragmentToSmiles(mol, list(ring_idxs),
                                                   isomericSmiles=False,
                                                   kekuleSmiles=True)

            if ring_smiles in self.ring_smiles_set:
                row_idx = self.ring_vocab.index[self.ring_vocab['name']
                                                == ring_smiles]
                row_idx = int(row_idx[0])
            else:
                row_idx = self.misc_ring_idx

            clusters.append(Cluster(row_idx, 'ring', ring_idxs))

        # add all remaining singular atoms
        leftover_atoms = set(range(mol.GetNumAtoms()))
        for cluster in clusters:
            leftover_atoms.difference_update(cluster.atom_idxs)

        for atom_idx in leftover_atoms:
            atomic_num = mol.GetAtomWithIdx(atom_idx).GetAtomicNum()
            clusters.append(Cluster(atomic_num, 'atom', (atom_idx,)))

        return clusters

    # noinspection PyMethodMayBeStatic
    def _make_inter_edges(self, clusters):

        c2atom_edges = [[], []]
        atom2c_edges = [[], []]

        for cluster_idx, cluster in enumerate(clusters):
            for atom_idx in cluster.atom_idxs:
                c2atom_edges[0].append(cluster_idx)
                c2atom_edges[1].append(atom_idx)

                atom2c_edges[0].append(atom_idx)
                atom2c_edges[1].append(cluster_idx)

        c2atom_edges = torch.tensor(c2atom_edges, dtype=torch.long)
        atom2c_edges = torch.tensor(atom2c_edges, dtype=torch.long)

        return c2atom_edges, atom2c_edges

    # noinspection PyMethodMayBeStatic
    def _make_intracluster_edges(self, raw_graph, clusters):

        edge_index = raw_graph['edge_index']

        edge_dict = {i: set() for i in range(raw_graph['num_nodes'])}
        for i, j in zip(edge_index[0], edge_index[1]):
            edge_dict[i].add(j)

        num_clusters = len(clusters)
        adj_matrix = [[0] * num_clusters for _ in range(num_clusters)]

        cluster_neighbours = []
        for cluster in clusters:
            neighbours = set()
            for atom_idx in cluster.atom_idxs:
                neighbours.add(atom_idx)
                neighbours.update(edge_dict[atom_idx])
            cluster_neighbours.append(neighbours)

        for i, j in itertools.combinations(range(num_clusters), r=2):
            ci, cj = clusters[i], clusters[j]

            if ci.atom_idxs & cj.atom_idxs:
                edge_weight = len(ci.atom_idxs & cj.atom_idxs) + 1
            elif cluster_neighbours[i] & cluster_neighbours[j]:
                edge_weight = 1
            else:
                continue

            adj_matrix[i][j] = edge_weight
            adj_matrix[j][i] = edge_weight

        # build spanning tree
        adj_matrix = csr_matrix(adj_matrix)
        span_tree = minimum_spanning_tree(adj_matrix, overwrite=True)
        adj_matrix = torch.from_numpy(span_tree.toarray()).long()
        adj_matrix = to_bidirectional(adj_matrix)

        # represent as sparse matrix
        adj_matrix = adj_matrix.to_sparse().coalesce()
        edge_index = adj_matrix.indices()
        edge_feats = adj_matrix.values()

        return edge_index, edge_feats


class Cluster:

    def __init__(self, vocab_id, cluster_type, atom_idxs):

        # for sanity
        if not isinstance(vocab_id, int):
            raise ValueError()

        self.vocab_id = vocab_id
        self.cluster_type_idx = ('fgroup', 'ring', 'atom').index(cluster_type)
        self.atom_idxs = frozenset(atom_idxs)

        self.features = [self.vocab_id, self.cluster_type_idx]


# Helper Method

def to_bidirectional(X):
    X_T = X.t()
    sym_sum = X + X_T
    X_min = torch.min(X, X_T)

    # noinspection PyTypeChecker
    return torch.where(X_min > 0, X_min, sym_sum)
