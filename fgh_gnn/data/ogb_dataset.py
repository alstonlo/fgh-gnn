import pathlib
import shutil

import dgl
import dgl.data.utils as data_utils
import numpy as np
import ogb.utils.url as url_utils
import pandas as pd
import torch
import tqdm
from ogb.io.read_graph_raw import read_csv_graph_raw
from rdkit import Chem

import fgh_gnn.utils as utils
from .graph_builder import FGroupHetGraphBuilder


class OGBPropPredDataset(dgl.data.DGLDataset):
    """Adapted from [1]

    References:
        [1] https://github.com/snap-stanford/ogb/blob/master/
            ogb/graphproppred/dataset_dgl.py
    """

    def __init__(self, name, root="dataset", min_count=10):

        self.min_count = min_count

        # important file paths
        root = pathlib.Path(root)

        self.original_root = root
        self.dir_name = "_".join(name.split("-")) + "_dgl"
        self.root = root / self.dir_name

        # reading meta-info
        meta_info_path = pathlib.Path(__file__).parent / "master.csv"
        self.meta_info = pd.read_csv(meta_info_path, index_col=0)

        if name not in self.meta_info:
            raise ValueError("Invalid dataset name.")

        self.download_name = self.meta_info[name]["download_name"]
        self.num_tasks = int(self.meta_info[name]["num tasks"])
        self.eval_metric = self.meta_info[name]["eval metric"]
        self.task_type = self.meta_info[name]["task type"]
        self.num_classes = self.meta_info[name]["num classes"]

        # to be assigned later
        self.vocab = None
        self.graphs = None
        self.labels = None

        super().__init__(
            name=name,
            url=self.meta_info[name]["url"],
            raw_dir=self.root,
            save_dir=self.root,
            # force_reload=True,  # TODO: remove me
            verbose=True
        )

    @property
    def raw_path(self):
        return self.root / 'raw'

    @property
    def save_path(self):
        return self.root / 'processed'

    @property
    def cache_paths(self):
        graph_cache_path = str(self.save_path / 'dgl_data_processed')
        vocab_cache_path = str(self.save_path / 'vocab.csv')
        return graph_cache_path, vocab_cache_path

    def download(self):
        if url_utils.decide_download(self.url):
            path = url_utils.download_url(self.url, self.original_root)
            path = pathlib.Path(path)

            url_utils.extract_zip(path, self.original_root)
            path.unlink()
            shutil.rmtree(self.root, ignore_errors=True)
            shutil.move(self.original_root / self.download_name, self.root)
        else:
            print("Stop download.")
            exit(-1)

    def process(self):

        # true for all chemistry-related datasets
        raw_graphs = read_csv_graph_raw(self.raw_path,
                                        add_inverse_edge=True,
                                        additional_node_files=[],
                                        additional_edge_files=[])
        labels = pd.read_csv(self.raw_path / "graph-label.csv.gz",
                             compression="gzip", header=None).values
        has_nan = np.isnan(labels).any()

        if "classification" in self.task_type:
            if has_nan:
                labels = torch.from_numpy(labels).to(torch.float32)
            else:
                labels = torch.from_numpy(labels).to(torch.long)
        else:
            labels = torch.from_numpy(labels).to(torch.float32)

        # build vocab
        self.vocab = self._build_vocab()

        self.graphs = []
        self.labels = labels

        builder = FGroupHetGraphBuilder(self.vocab)
        for raw_g in tqdm.tqdm(raw_graphs, desc="Building FG-Graphs"):
            g = builder.build_fgroup_heterograph(raw_g)
            self.graphs.append(g)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.graphs[idx], self.labels[idx]

        elif torch.is_tensor(idx) and (idx.dtype == torch.long):
            if idx.dim() == 0:
                return self.graphs[idx], self.labels[idx]
            elif idx.dim() == 1:
                return data_utils.Subset(self, idx.cpu())

        raise IndexError('Only integers and long are valid indices (got {}).'
                         .format(type(idx).__name__))

    def __len__(self):
        return len(self.graphs)

    def save(self):
        graph_cache_path, vocab_cache_path = self.cache_paths

        data_utils.save_graphs(graph_cache_path,
                               self.graphs,
                               labels={'labels': self.labels})
        self.vocab.to_csv(vocab_cache_path)

    def load(self):
        graph_cache_path, vocab_cache_path = self.cache_paths

        self.graphs, label_dict = data_utils.load_graphs(graph_cache_path)
        self.labels = label_dict['labels']
        self.vocab = pd.read_csv(vocab_cache_path)

    def has_cache(self):
        return all(pathlib.Path(p).exists() for p in self.cache_paths)

    def get_idx_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info[self.name]["split"]

        path = self.root / "split" / split_type
        train_idx = pd.read_csv(path / "train.csv.gz",
                                compression="gzip",
                                header=None).values.T[0]
        valid_idx = pd.read_csv(path / "valid.csv.gz",
                                compression="gzip",
                                header=None).values.T[0]
        test_idx = pd.read_csv(path / "test.csv.gz",
                               compression="gzip",
                               header=None).values.T[0]

        return {"train": torch.tensor(train_idx, dtype=torch.long),
                "valid": torch.tensor(valid_idx, dtype=torch.long),
                "test": torch.tensor(test_idx, dtype=torch.long)}

    def _build_vocab(self):

        # analyze functional groups and rings
        split_idx = self.get_idx_split()
        train_idx = set(i.item() for i in split_idx["train"])

        train_smiles = pd.read_csv(
            self.root / "mapping" / "mol.csv.gz",
            skiprows=(lambda x: (x != 0) and (x not in train_idx)),
        )
        train_smiles = train_smiles['smiles'].tolist()

        fgroups_and_rings = utils.analyze_fgroups_and_rings(train_smiles)

        # save for future
        fgroups_and_rings.to_csv(self.save_path / 'fgroup_and_rings.csv')

        # extract vocab from analysis
        vocab = fgroups_and_rings[fgroups_and_rings['count'] >= self.min_count]

        atom_data = []  # add atom data
        for atomic_num in range(0, 119):
            atom = Chem.Atom(atomic_num)
            atom_data.append({'name': atom.GetSymbol(), 'type': 'atom'})

        vocab = pd.concat([pd.DataFrame(atom_data), vocab], ignore_index=True)
        vocab = vocab.append({'name': 'misc_ring', 'type': 'ring'},
                             ignore_index=True)

        return vocab
