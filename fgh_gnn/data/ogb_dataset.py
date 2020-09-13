import numpy as np
import ogb.utils.url as url_utils
import pandas as pd
import shutil
import torch
from ogb.io.read_graph_pyg import read_csv_graph_pyg
from pathlib import Path
from rdkit import Chem
from torch_geometric.data import InMemoryDataset

import fgh_gnn.utils as utils
from .graph_builder import FGroupHetGraphBuilder


class OGBPropPredDataset(InMemoryDataset):

    def __init__(self, name, root, min_count):

        self.name = name
        self.min_count = min_count

        # important file paths
        root = Path(root)

        self.original_root = root
        self.dir_name = "_".join(name.split("-")) + "_pyg"
        self.root = root / self.dir_name

        # reading meta-info
        meta_info_path = Path(__file__).parent / "master.csv"
        self.meta_info = pd.read_csv(meta_info_path, index_col=0)

        if self.name not in self.meta_info:
            raise ValueError("Invalid dataset name.")

        self.download_name = self.meta_info[self.name]["download_name"]
        self.num_tasks = int(self.meta_info[self.name]["num tasks"])
        self.eval_metric = self.meta_info[self.name]["eval metric"]
        self.task_type = self.meta_info[self.name]["task type"]
        self.__num_classes__ = int(self.meta_info[self.name]["num classes"])

        super().__init__(self.root)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.vocab = pd.read_csv(self.processed_paths[1])

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

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        file_names = ["edge"]
        if self.meta_info[self.name]["has_node_attr"] == "True":
            file_names.append("node-feat")
        if self.meta_info[self.name]["has_edge_attr"] == "True":
            file_names.append("edge-feat")
        return [file_name + ".csv.gz" for file_name in file_names]

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt", \
               'vocab.csv', \
               'fgroup_and_rings.csv'

    def download(self):
        url = self.meta_info[self.name]["url"]
        if url_utils.decide_download(url):
            path = url_utils.download_url(url, self.original_root)
            path = Path(path)

            url_utils.extract_zip(path, self.original_root)
            path.unlink()
            shutil.rmtree(self.root, ignore_errors=True)
            shutil.move(self.original_root / self.download_name, self.root)
        else:
            print("Stop download.")
            exit(-1)

    def process(self):

        # true for all chemistry-related datasets
        data_list = read_csv_graph_pyg(self.raw_dir,
                                       add_inverse_edge=True,
                                       additional_node_files=[],
                                       additional_edge_files=[])
        graph_label = pd.read_csv(Path(self.raw_dir) / "graph-label.csv.gz",
                                  compression="gzip",
                                  header=None).values
        has_nan = np.isnan(graph_label).any()

        for i, g in enumerate(data_list):
            if "classification" in self.task_type:
                if has_nan:
                    g.y = torch.from_numpy(graph_label[i])
                    g.y = g.y.view(1, -1).to(torch.float32)
                else:
                    g.y = torch.from_numpy(graph_label[i]).view(1, -1)
                    g.y = g.y.view(1, -1).to(torch.long)
            else:
                g.y = torch.from_numpy(graph_label[i])
                g.y = g.y.view(1, -1).to(torch.float32)

        # build vocab and graphs
        vocab = self._build_vocab()

        graph_builder = FGroupHetGraphBuilder(vocab)
        data_list = [graph_builder(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])
        vocab.to_csv(self.processed_paths[1])

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

        # extract vocab from analysis
        vocab = fgroups_and_rings[fgroups_and_rings['count'] >= self.min_count]

        atom_data = []  # add atom data
        for atomic_num in range(0, 119):
            atom = Chem.Atom(atomic_num)
            atom_data.append({'name': atom.GetSymbol(), 'type': 'atom'})

        vocab = pd.concat([pd.DataFrame(atom_data), vocab], ignore_index=True)
        vocab = vocab.append({'name': 'misc_ring', 'type': 'ring'},
                             ignore_index=True)

        # save for future
        fgroups_and_rings.to_csv(self.processed_paths[2])

        return vocab
