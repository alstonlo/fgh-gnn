import itertools
import torch
import torch_geometric as pyg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from fgh_gnn.utils import FGROUP_MOLS, get_ring_fragments, pyg_graph_to_mol


class FGroupHetGraph(pyg.data.Data):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # To be assigned later
        self.x_cluster = None
        self.c2c_edge_index = None
        self.c2c_edge_attr = None
        self.num_clusters = None

        self.c2atom_edge_index = None
        self.atom2c_edge_index = None

    def __inc__(self, key, value):
        if key == 'c2c_edge_index':
            return self.x_cluster.size(0)
        elif key == 'atom2c_edge_index':
            return torch.tensor([[self.x.size(0)], [self.x_cluster.size(0)]])
        elif key == 'c2atom_edge_index':
            return torch.tensor([[self.x_cluster.size(0)], [self.x.size(0)]])
        else:
            return super().__inc__(key, value)


class Cluster:

    cluster_types = ('fgroup', 'ring', 'atom')
    max_cluster_size = 60

    def __init__(self, cluster_type, members):
        self.cluster_type_idx = self.cluster_types.index(cluster_type)
        self.members = frozenset(members)

        if len(members) > self.max_cluster_size:
            raise ValueError(f"cluster too large (size {len(members)}).")

        self.features = [self.cluster_type_idx, len(members)]


def build_fgroup_heterograph(data):
    # build tree
    mol = pyg_graph_to_mol(data)
    clusters = make_clusters(mol)
    cluster_attr = torch.tensor([c.features for c in clusters],
                                dtype=torch.long)

    c2atom_edges, atom2c_edges = make_inter_edges(clusters)
    c2c_edges, c2c_edge_attr = make_intracluster_edges(data, clusters)

    # build heterograph
    g = FGroupHetGraph(**{k: v for k, v in data})

    g.x_cluster = cluster_attr
    g.c2c_edge_index = c2c_edges
    g.c2c_edge_attr = c2c_edge_attr
    g.num_clusters = len(clusters)

    g.c2atom_edge_index = c2atom_edges
    g.atom2c_edge_index = atom2c_edges

    return g


def make_clusters(mol):
    clusters = []

    # add all functional groups
    for fgroup_query in FGROUP_MOLS.values():
        matches = mol.GetSubstructMatches(fgroup_query)
        for match_idxs in matches:
            clusters.append(Cluster('fgroup', match_idxs))

    # add all rings
    for ring_idxs in get_ring_fragments(mol):
        clusters.append(Cluster('ring', ring_idxs))

    # add all atoms as clusters of size 1
    for atom_idx in range(mol.GetNumAtoms()):
        clusters.append(Cluster('atom', (atom_idx,)))

    # remove all clusters such that:
    #   * no cluster is a subset of another
    #   * no cluster shares 3+ atoms with another cluster
    clusters.sort(key=lambda x: len(x.members), reverse=True)

    cleaned_clusters = []
    for c in clusters:

        add_c = True
        for o in cleaned_clusters:
            if (c.members <= o.members) or (len(c.members & o.members) >= 3):
                add_c = False
                break

        if add_c:
            cleaned_clusters.append(c)

    return cleaned_clusters


def make_inter_edges(clusters):
    c2atom_edges = [[], []]
    atom2c_edges = [[], []]

    for cluster_idx, cluster in enumerate(clusters):
        for atom_idx in cluster.members:
            c2atom_edges[0].append(cluster_idx)
            c2atom_edges[1].append(atom_idx)

            atom2c_edges[0].append(atom_idx)
            atom2c_edges[1].append(cluster_idx)

    c2atom_edges = torch.tensor(c2atom_edges, dtype=torch.long)
    atom2c_edges = torch.tensor(atom2c_edges, dtype=torch.long)

    return c2atom_edges, atom2c_edges


def make_intracluster_edges(data, clusters):
    edge_index = data.edge_index.tolist()

    edge_dict = {i: set() for i in range(data.num_nodes)}
    for i, j in zip(edge_index[0], edge_index[1]):
        edge_dict[i].add(j)

    num_clusters = len(clusters)
    adj_matrix = [[0] * num_clusters for _ in range(num_clusters)]

    cluster_neighbours = []
    for cluster in clusters:
        neighbours = set()
        for atom_idx in cluster.members:
            neighbours.add(atom_idx)
            neighbours.update(edge_dict[atom_idx])
        cluster_neighbours.append(neighbours)

    for i, j in itertools.combinations(range(num_clusters), r=2):
        ci, cj = clusters[i], clusters[j]

        if ci.members & cj.members:
            edge_weight = len(ci.members & cj.members) + 1
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
    edge_attr = adj_matrix.values()

    # since we added 1 previously to the edge weights
    edge_attr = edge_attr.unsqueeze(1) - 1

    return edge_index, edge_attr


# Helper Method

def to_bidirectional(X):
    X_T = X.t()
    sym_sum = X + X_T
    X_min = torch.min(X, X_T)

    return torch.where(X_min > 0, X_min, sym_sum)
