import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
import atom3d.util.graph as gr
from atom3d.util.transforms import prot_graph_transform, mol_graph_transform, PairedGraphTransform
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data, Batch, InMemoryDataset
from torch_geometric.data.separate import separate
import subprocess
import scipy
import scipy.spatial as ss
import os.path as osp
import copy
from joblib import Parallel, delayed
from pysign.data import from_pyg
from .registry import DatasetRegistry
from .pip import neighbors as nb


def combine_graphs(graph1, graph2, edges_between=True, edges_between_dist=4.5):
    """Combine two graphs into one, optionally adding edges between the two graphs using :func:`atom3d.util.graph.edges_between_graphs`.
    Node features are concatenated in the feature dimension, to distinguish which nodes came from which graph.

    :param graph1: One of the graphs to be combined, in the format returned by :func:`atom3d.util.graph.prot_df_to_graph` or :func:`atom3d.util.graph.mol_df_to_graph`.
    :type graph1: Tuple
    :param graph2: The other graph to be combined, in the format returned by :func:`atom3d.util.graph.prot_df_to_graph` or :func:`atom3d.util.graph.mol_df_to_graph`.
    :type graph2: Tuple
    :param edges_between: Indicates whether to add new edges between graphs, defaults to True.
    :type edges_between: bool, optional
    :param edges_between_dist: Distance cutoff in Angstroms for adding edges between graphs, defaults to 4.5.
    :type edges_between_dist: float, optional
    :return: Tuple containing \n
        - node_feats (torch.FloatTensor): Features for each node in the combined graph, concatenated along the feature dimension.\n
        - edges (torch.LongTensor): Edges of combined graph in COO format, including edges from two input graphs and edges between them, if specified.\n
        - edge_weights (torch.FloatTensor): Concatenated edge features from two input graphs and edges between them, if specified.\n
        - node_pos (torch.FloatTensor): x-y-z coordinates of each node in combined graph.
    :rtype: Tuple
    """
    node_feats1, edges1, edge_feats1, pos1 = graph1.x, graph1.edge_index, graph1.edge_attr, graph1.pos
    node_feats2, edges2, edge_feats2, pos2 = graph2.x, graph2.edge_index, graph2.edge_attr, graph2.pos

    dummy_node_feats1 = torch.zeros(pos1.shape[0], node_feats2.shape[1])
    dummy_node_feats2 = torch.zeros(pos2.shape[0], node_feats1.shape[1])
    node_feats1 = torch.cat((node_feats1, dummy_node_feats1), dim=1)
    node_feats2 = torch.cat((dummy_node_feats2, node_feats2), dim=1)

    node_mask = torch.cat([torch.zeros(pos1.shape[0]), torch.ones(pos2.shape[0])]).long()

    edges2 += pos1.shape[0]

    node_pos = torch.cat((pos1, pos2), dim=0)
    node_feats = torch.cat((node_feats1, node_feats2), dim=0)

    if edges_between:
        edges_between, edge_feats_between = edges_between_graphs(pos1, pos2)
        edge_feats = torch.cat((edge_feats1, edge_feats2, edge_feats_between), dim=0)
        edges = torch.cat((edges1, edges2, edges_between), dim=1)
    else:
        edge_feats = torch.cat((edge_feats1, edge_feats2), dim=0)
        edges = torch.cat((edges1, edges2), dim=1)

    return node_feats, edges, edge_feats, node_pos, node_mask


def edges_between_graphs(pos1, pos2, dist=4.5):
    """calculates edges between nodes in two separate graphs using a specified cutoff distance.

    :param pos1: x-y-z node coordinates from Graph 1
    :type pos1: torch.FloatTensor or numpy.ndarray
    :param pos2: x-y-z node coordinates from Graph 2
    :type pos2: torch.FloatTensor or numpy.ndarray
    :return: Tuple containing\n
        - edges (torch.LongTensor): Edges between two graphs, in COO format.\n
        - edge_weights (torch.FloatTensor): Edge weights between two graphs.\n
    :rtype: Tuple
    """
    tree1 = ss.KDTree(pos1)
    tree2 = ss.KDTree(pos2)
    res = tree1.query_ball_tree(tree2, r=dist)
    edges = []
    edge_weights = []
    for i, contacts in enumerate(res):
        if len(contacts) == 0:
            continue
        for j in contacts:
            edges.append((i, j + pos1.shape[0]))
            edges.append((j + pos1.shape[0], i))
            d = 1.0 / (np.linalg.norm(pos1[i] - pos2[j]) + 1e-5)
            edge_weights.append(d)
            edge_weights.append(d)

    edges = torch.LongTensor(edges).t().contiguous()
    edge_weights = torch.FloatTensor(edge_weights).view(-1)
    return edges, edge_weights


# adapted from atom3d.examples.lep.gnn.data
class GNNTransformLEP(object):
    def __init__(self):
        self.atom_keys = ['atoms_active', 'atoms_inactive']
        self.label_key = 'label'

    def __call__(self, item):
        # transform protein and/or pocket to PTG graphs
        item = prot_graph_transform(item, atom_keys=self.atom_keys, label_key=self.label_key)
        attrs = ['x', 'edge_index', 'edge_attr', 'y', 'pos']
        transform = from_pyg(attrs)
        return transform(item[self.atom_keys[0]]), transform(item[self.atom_keys[1]])


class CollaterLEP(object):
    """To be used with pre-computed graphs and atom3d.datasets.PTGDataset"""

    def __init__(self):
        pass

    def __call__(self, data_list):
        batch_1 = Batch.from_data_list([d[0] for d in data_list])
        batch_2 = Batch.from_data_list([d[1] for d in data_list])
        return batch_1, batch_2


class GNNTransformLBA(object):
    def __init__(self, pocket_only=True):
        self.pocket_only = pocket_only

    def __call__(self, item):
        # transform protein and/or pocket to PTG graphs
        if self.pocket_only:
            item = prot_graph_transform(item, atom_keys=['atoms_pocket'], label_key='scores')
        else:
            item = prot_graph_transform(item, atom_keys=['atoms_protein', 'atoms_pocket'], label_key='scores')
        # transform ligand into PTG graph
        item = mol_graph_transform(item, 'atoms_ligand', 'scores', use_bonds=True, onehot_edges=False)
        node_feats, edges, edge_feats, node_pos, node_mask = combine_graphs(item['atoms_pocket'], item['atoms_ligand'])
        combined_graph = Data(node_feats, edges, edge_feats, y=item['scores']['neglog_aff'], pos=node_pos,
                              instance=node_mask)
        attrs = ['x', 'edge_index', 'edge_attr', 'y', 'pos', 'instance']
        combined_graph = from_pyg(attrs)(combined_graph)
        return combined_graph


# handling protein interface prediction (PIP) dataset
class GNNTransformPIP(object):

    """
    The processing logic are mostly the same as the official implementation at:
    https://github.com/drorlab/atom3d/blob/master/examples/ppi/gnn/data.py
    """

    def __init__(self, residue_level=False, backbone_only=False):
        self.residue_level = residue_level
        self.backbone_only = backbone_only

    def __call__(self, item):
        neighbors = item['atoms_neighbors']
        pairs = item['atoms_pairs']

        graph_list1, graph_list2 = [], []
        for i, (ensemble_name, target_df) in enumerate(pairs.groupby(['ensemble'])):
            sub_names, (bound1, bound2, _, _) = nb.get_subunits(target_df)
            positives = neighbors[neighbors.ensemble0 == ensemble_name]
            negatives = nb.get_negatives(positives, bound1, bound2)
            negatives['label'] = 0
            labels = self.create_labels(positives, negatives, num_pos=10, neg_pos_ratio=1)

            for index, row in labels.iterrows():
                label = float(row['label'])
                chain_res1 = row[['chain0', 'residue0']].values
                chain_res2 = row[['chain1', 'residue1']].values
                graph1 = self.df_to_graph(bound1, chain_res1, label)
                graph2 = self.df_to_graph(bound2, chain_res2, label)
                if (graph1 is None) or (graph2 is None):
                    continue
                graph_list1.append(graph1)
                graph_list2.append(graph2)
        
        return graph_list1, graph_list2

    def create_labels(self, positives, negatives, num_pos, neg_pos_ratio):
        frac = min(1, num_pos / positives.shape[0])
        positives = positives.sample(frac=frac)
        n = positives.shape[0] * neg_pos_ratio
        negatives = negatives.sample(n, random_state=0, axis=0)
        labels = pd.concat([positives, negatives])[['chain0', 'residue0', 'chain1', 'residue1', 'label']]
        return labels

    def df_to_graph(self, struct_df, chain_res, label):
        """
        Extracts atoms within 30A of CA atom and computes graph
        """
        chain, resnum = chain_res
        res_df = struct_df[(struct_df.chain == chain) & (struct_df.residue == resnum)]
        if 'CA' not in res_df.name.tolist():
            return None
        ca_pos = res_df[res_df['name'] == 'CA'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]

        kd_tree = scipy.spatial.KDTree(struct_df[['x', 'y', 'z']].to_numpy())
        graph_pt_idx = kd_tree.query_ball_point(ca_pos, r=30.0, p=2.0)
        graph_df = struct_df.iloc[graph_pt_idx].reset_index(drop=True)
        ca_idx = np.where((graph_df.chain == chain) & (graph_df.residue == resnum) & (graph_df.name == 'CA'))[0]
        if not len(ca_idx) > 0:
            return None

        node_feats, edge_index, edge_feats, pos = gr.prot_df_to_graph(graph_df)
        data = Data(node_feats, edge_index, edge_feats, y=label, pos=pos)
        data.ca_idx = torch.LongTensor(ca_idx)
        data.n_nodes = data.num_nodes
        return data


def iter_samples(lmdb, idx):
    return lmdb[idx]


def get_preprocessed_list(lmdb, num_workers):
    res = []
    if num_workers == 1:
        for i, item in enumerate(tqdm(lmdb)):
            res.append(item)
        return res
    res = Parallel(n_jobs=num_workers, backend='threading')(
        delayed(iter_samples)(lmdb, idx) for idx in tqdm(range(len(lmdb))))
    return res


@DatasetRegistry.register_dataset('atom3d')
class Atom3DDataset(InMemoryDataset):
    """Machine learning of accurate energy-conserving molecular force fields (Chmiela et al. 2017)
    This class provides functionality for loading MD trajectories from the original dataset, not the revised versions.
    See http://www.quantum-machine.org/gdml/#datasets for details.
    """

    multi_graph_tasks = ['lep', 'pip']
    one_to_many_tasks = ['pip']

    def __init__(self, root, task, split, transform=None, pre_transform=None, dataset_arg=None, num_workers=1):

        self.task = task.lower()
        self.split = split
        self.dataset_arg = dataset_arg
        self.num_workers = num_workers
        if not pre_transform:
            pre_transform = self.get_pre_transform()
        super(Atom3DDataset, self).__init__(root, transform, pre_transform)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        if self.task in Atom3DDataset.multi_graph_tasks:
            self.data1, self.data2 = self.data
            self.slices1, self.slices2 = self.slices

    def len(self):
        if self.task in Atom3DDataset.multi_graph_tasks:
            slices = self.slices1
        else:
            slices = self.slices
        return len(slices[list(slices.keys())[0]]) - 1

    def sep(self, data, slices, idx):
        data = separate(
            cls=data.__class__,
            batch=data,
            idx=idx,
            slice_dict=slices,
            decrement=False,
        )
        return data

    def get(self, idx: int):

        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]

        if self._data_list[idx] is not None:
            data = copy.copy(self._data_list[idx])

        else:
            if self.task in Atom3DDataset.multi_graph_tasks:
                data1 = self.sep(self.data1, self.slices1, idx)
                data2 = self.sep(self.data2, self.slices2, idx)
                data = (data1, data2)
            else:
                data = self.sep(self.data, self.slices, idx)

            self._data_list[idx] = copy.copy(data)

        return data

    def get_pre_transform(self):
        name = self.task
        if name == "lep":
            pre_transform = GNNTransformLEP()
        elif name == "lba":
            pre_transform = GNNTransformLBA()
        elif name == "pip":
            pre_transform = GNNTransformPIP()
        else:
            raise NotImplementedError('Unknown task', name)
        return pre_transform

    @property
    def lmdb_file_dir(self):
        file_dir = self.raw_file_names[:-7]
        return osp.join(self.raw_dir, file_dir, 'data')

    @property
    def raw_dir(self):
        return osp.join(self.root, self.task, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.task, 'processed')

    @property
    def download_url(self):
        name, out_path, split = self.task, self.root, self.dataset_arg

        if name == 'smp':
            if split is None:
                link = 'https://zenodo.org/record/4911142/files/SMP-raw.tar.gz?download=1'
            elif split == 'random':
                link = 'https://zenodo.org/record/4911142/files/SMP-random.tar.gz?download=1'
            else:
                print(f'specified split {split} not available. Possible values are "random".')
                link = ""
        elif name == 'pip':
            if split is None:
                link = 'https://zenodo.org/record/4911102/files/PPI-raw.tar.gz?download=1'
            elif split == 'DIPS':
                link = 'https://zenodo.org/record/4911102/files/PPI-DIPS-split.tar.gz?download=1'
            else:
                print(f'specified split {split} not available. Possible values are "DIPS".')
                link = ""
        elif name == 'res':
            if split is None:
                link = 'https://zenodo.org/record/5026743/files/RES-raw.tar.gz?download=1'
            elif split == 'cath-topology':
                link = 'https://zenodo.org/record/5026743/files/RES-split-by-cath-topology.tar.gz?download=1'
            else:
                print(f'specified split {split} not available. Possible values are "cath-topology".')
                link = ""
        elif name == 'msp':
            if split is None:
                link = 'https://zenodo.org/record/4962515/files/MSP-raw.tar.gz?download=1'
            elif split == 'sequence-identity-30':
                link = 'https://zenodo.org/record/4962515/files/MSP-split-by-sequence-identity-30.tar.gz?download=1'
            else:
                print(f'specified split {split} not available. Possible values are "sequence-identity-30".')
                link = ""
        elif name == 'lba':
            if split is None:
                link = 'https://zenodo.org/record/4914718/files/LBA-raw.tar.gz?download=1'
            elif split == 'sequence-identity-30':
                link = 'https://zenodo.org/record/4914718/files/LBA-split-by-sequence-identity-30.tar.gz?download=1'
            elif split == 'sequence-identity-60':
                link = 'https://zenodo.org/record/4914718/files/LBA-split-by-sequence-identity-60.tar.gz?download=1'
            else:
                print(
                    f'specified split {split} not available. Possible values are "sequence-identity-30", "sequence-identity-60".')
                link = ""
        elif name == 'lep':
            if split is None:
                link = 'https://zenodo.org/record/4914734/files/LEP-raw.tar.gz?download=1-rri'
            elif split == 'protein':
                link = 'https://zenodo.org/record/4914734/files/LEP-split-by-protein.tar.gz?download=1'
            else:
                print(f'specified split {split} not available. Possible values are "protein".')
                link = ""
        elif name == 'psr':
            if split is None:
                link = 'https://zenodo.org/record/4915648/files/PSR-raw.tar.gz?download=1'
            elif split == 'year':
                link = 'https://zenodo.org/record/4915648/files/PSR-split-by-year.tar.gz?download=1'
            else:
                print(f'specified split {split} not available. Possible values are "year".')
                link = ""
        elif name == 'rsr':
            if split is None:
                link = 'https://zenodo.org/record/4961085/files/RSR-raw.tar.gz?download=1'
            elif split == 'year':
                link = 'https://zenodo.org/record/4961085/files/RSR-candidates-split-by-time.tar.gz?download=1'
            else:
                print(f'specified split {split} not available. Possible values are "year".')
                link = ""
        else:
            print('Invalid dataset name specified. Possible values are {SMP, PIP, RES, MSP, LBA, LEP, PSR, RSR}')
            link = ""
        return link

    @property
    def raw_file_names(self):
        assert self.download_url != ""
        file_name = self.download_url.split('?')[0].split('/')[-1][4:]
        return file_name

    @property
    def processed_file_names(self):
        return [f'{split}.pt' for split in ['train', 'val', 'test']]

    def download(self):
        """Download an ATOM3D dataset in LMDB format. Available datasets are SMP, PIP, RES, MSP, LBA, LEP, PSR, RSR.
        Please see `FAQ <datasets target>`_ or `atom3d.ai <atom3d.ai>`_ for more details on each dataset.

        :param name: Three-letter code for dataset (not case-sensitive).
        :type name: str
        :param out_path: Path to directory in which to save downloaded dataset.
        :type out_path: str
        :param split: name of split data to download in LMDB format. Defaults to None, in which case raw (unsplit)
        dataset is downloaded. Please use :func:`download_split_indices` to get pre-computed split indices for raw datasets.
        :type split: str
        """
        link = self.download_url
        name = self.raw_file_names
        out_path = self.raw_dir
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        # cmd = f"wget {link} -O {out_path}/{name}"
        cmd = f"wget {link} -O {out_path}/{name} --no-check-certificate"   # insecure connection
        subprocess.call(cmd, shell=True)
        cmd2 = f"tar xzvf {out_path}/{name} -C {out_path}"
        subprocess.call(cmd2, shell=True)

    def process(self):
        data_dir = self.lmdb_file_dir
        save_dir = self.processed_dir
        one_to_many = self.task in Atom3DDataset.one_to_many_tasks
        if self.task in Atom3DDataset.multi_graph_tasks:
            for split in ['train', 'val', 'test']:
                print(f"Preprocessing {split} data for {self.task} task ...")
                dataset = LMDBDataset(os.path.join(data_dir, split), transform=self.pre_transform)
                samples1, samples2 = [], []
                if one_to_many:
                    for i, item_list in enumerate(tqdm(dataset)):
                        item_list1, item_list2 = item_list
                        samples1.extend(item_list1)
                        samples2.extend(item_list2)
                else:
                    for i, item in enumerate(tqdm(dataset)):
                        item1, item2 = item
                        samples1.append(item1)
                        samples2.append(item2)
                data1, slices1 = self.collate(samples1)
                data2, slices2 = self.collate(samples2)
                torch.save(((data1, data2), (slices1, slices2)), osp.join(save_dir, f'{split}.pt'))
        else:
            for split in ['train', 'val', 'test']:
                print(f"Preprocessing {split} data for {self.task} task ...")
                dataset = LMDBDataset(os.path.join(data_dir, split), transform=self.pre_transform)
                samples = []
                if one_to_many:
                    for i, item_list in enumerate(tqdm(dataset)):
                        samples.extend(item_list)
                else:
                    for i, item in enumerate(tqdm(dataset)):
                        samples.append(item)
                # samples = get_preprocessed_list(dataset,self.num_workers)
                data, slices = self.collate(samples)
                torch.save((data, slices), osp.join(save_dir, f'{split}.pt'))
        print("Preprocess done!")


if __name__ == '__main__':
    dataset = Atom3DDataset('/data/private/kxz/cached_datasets/atom3d', 'pip', 'train', dataset_arg='protein')
