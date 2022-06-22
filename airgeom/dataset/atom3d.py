import numpy as np
import os
import torch
from tqdm import tqdm
from atom3d.util.transforms import prot_graph_transform, mol_graph_transform, PairedGraphTransform
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data, Batch, DataLoader
import atom3d.util.graph as gr
import subprocess
import scipy.spatial as ss


def combine_graphs(graph1, graph2, edges_between=True, edges_between_dist=4.5):
    """Combine two graphs into one, optionally adding edges between the two graphs using :func:`atom3d.util.graph.edges_between_graphs`. Node features are concatenated in the feature dimension, to distinguish which nodes came from which graph.

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

    return node_feats, edges, edge_feats, node_pos


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
    def __init__(self, atom_keys, label_key):
        self.atom_keys = atom_keys
        self.label_key = label_key

    def __call__(self, item):
        # transform protein and/or pocket to PTG graphs
        item = prot_graph_transform(item, atom_keys=self.atom_keys, label_key=self.label_key)

        return item


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
        node_feats, edges, edge_feats, node_pos = gr.combine_graphs(item['atoms_pocket'], item['atoms_ligand'])
        combined_graph = Data(node_feats, edges, edge_feats, y=item['scores']['neglog_aff'], pos=node_pos)
        return combined_graph


def preprocess_lba():
    # TODO: save dir data dir seqid -> args
    seqid = 30  # 30 or 60
    save_dir = '/scratch/users/aderry/atom3d/lba_' + str(seqid)
    data_dir = f'/scratch/users/raphtown/atom3d_mirror/lmdb/LBA/splits/split-by-sequence-identity-{seqid}/data'
    os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)
    train_dataset = LMDBDataset(os.path.join(data_dir, 'train'), transform=GNNTransformLBA())
    val_dataset = LMDBDataset(os.path.join(data_dir, 'val'), transform=GNNTransformLBA())
    test_dataset = LMDBDataset(os.path.join(data_dir, 'test'), transform=GNNTransformLBA())

    print('processing train dataset...')
    for i, item in enumerate(tqdm(train_dataset)):
        torch.save(item, os.path.join(save_dir, 'train', f'data_{i}.pt'))

    print('processing validation dataset...')
    for i, item in enumerate(tqdm(val_dataset)):
        torch.save(item, os.path.join(save_dir, 'val', f'data_{i}.pt'))

    print('processing test dataset...')
    for i, item in enumerate(tqdm(test_dataset)):
        torch.save(item, os.path.join(save_dir, 'test', f'data_{i}.pt'))


def preprocess_lep():
    save_dir = '/scratch/users/aderry/atom3d/lep'
    data_dir = '/scratch/users/raphtown/atom3d_mirror/lmdb/LEP/splits/split-by-protein/data'
    os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)
    transform = PairedGraphTransform('atoms_active', 'atoms_inactive', label_key='label')
    train_dataset = LMDBDataset(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = LMDBDataset(os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = LMDBDataset(os.path.join(data_dir, 'test'), transform=transform)

    print('processing train dataset...')
    for i, item in enumerate(tqdm(train_dataset)):
        torch.save(item, os.path.join(save_dir, 'train', f'data_{i}.pt'))

    print('processing validation dataset...')
    for i, item in enumerate(tqdm(val_dataset)):
        torch.save(item, os.path.join(save_dir, 'val', f'data_{i}.pt'))

    print('processing test dataset...')
    for i, item in enumerate(tqdm(test_dataset)):
        torch.save(item, os.path.join(save_dir, 'test', f'data_{i}.pt'))


def download_dataset(name, out_path, split=None):
    """Download an ATOM3D dataset in LMDB format. Available datasets are SMP, PIP, RES, MSP, LBA, LEP, PSR, RSR. Please see `FAQ <datasets target>`_ or `atom3d.ai <atom3d.ai>`_ for more details on each dataset.

    :param name: Three-letter code for dataset (not case-sensitive).
    :type name: str
    :param out_path: Path to directory in which to save downloaded dataset.
    :type out_path: str
    :param split: name of split data to download in LMDB format. Defaults to None, in which case raw (unsplit) dataset is downloaded. Please use :func:`download_split_indices` to get pre-computed split indices for raw datasets.
    :type split: str
    """

    name = name.lower()
    if name == 'smp':
        if split is None:
            link = 'https://zenodo.org/record/4911142/files/SMP-raw.tar.gz?download=1'
        elif split == 'random':
            link = 'https://zenodo.org/record/4911142/files/SMP-random.tar.gz?download=1'
        else:
            print(f'specified split {split} not available. Possible values are "random".')
            return
    elif name == 'ppi':
        if split is None:
            link = 'https://zenodo.org/record/4911102/files/PPI-raw.tar.gz?download=1'
        elif split == 'DIPS':
            link = 'https://zenodo.org/record/4911102/files/PPI-DIPS-split.tar.gz?download=1'
        else:
            print(f'specified split {split} not available. Possible values are "DIPS".')
            return
    elif name == 'res':
        if split is None:
            link = 'https://zenodo.org/record/5026743/files/RES-raw.tar.gz?download=1'
        elif split == 'cath-topology':
            link = 'https://zenodo.org/record/5026743/files/RES-split-by-cath-topology.tar.gz?download=1'
        else:
            print(f'specified split {split} not available. Possible values are "cath-topology".')
            return
    elif name == 'msp':
        if split is None:
            link = 'https://zenodo.org/record/4962515/files/MSP-raw.tar.gz?download=1'
        elif split == 'sequence-identity-30':
            link = 'https://zenodo.org/record/4962515/files/MSP-split-by-sequence-identity-30.tar.gz?download=1'
        else:
            print(f'specified split {split} not available. Possible values are "sequence-identity-30".')
            return
    elif name == 'lba':
        if split is None:
            link = 'https://zenodo.org/record/4914718/files/LBA-raw.tar.gz?download=1'
        elif split == 'sequence-identity-30':
            link = 'https://zenodo.org/record/4914718/files/LBA-split-by-sequence-identity-30.tar.gz?download=1'
        elif split == 'sequence-identity-60':
            link = 'https://zenodo.org/record/4914718/files/LBA-split-by-sequence-identity-60.tar.gz?download=1'
        else:
            print(f'specified split {split} not available. Possible values are "sequence-identity-30", "sequence-identity-60".')
            return
    elif name == 'lep':
        if split is None:
            link = 'https://zenodo.org/record/4914734/files/LEP-raw.tar.gz?download=1-rri'
        elif split == 'protein':
            link = 'https://zenodo.org/record/4914734/files/LEP-split-by-protein.tar.gz?download=1'
        else:
            print(f'specified split {split} not available. Possible values are "protein".')
            return
    elif name == 'psr':
        if split is None:
            link = 'https://zenodo.org/record/4915648/files/PSR-raw.tar.gz?download=1'
        elif split == 'year':
            link = 'https://zenodo.org/record/4915648/files/PSR-split-by-year.tar.gz?download=1'
        else:
            print(f'specified split {split} not available. Possible values are "year".')
            return
    elif name == 'rsr':
        if split is None:
            link = 'https://zenodo.org/record/4961085/files/RSR-raw.tar.gz?download=1'
        elif split == 'year':
            link = 'https://zenodo.org/record/4961085/files/RSR-candidates-split-by-time.tar.gz?download=1'
        else:
            print(f'specified split {split} not available. Possible values are "year".')
            return
    else:
        print('Invalid dataset name specified. Possible values are {SMP, PIP, RES, MSP, LBA, LEP, PSR, RSR}')

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    cmd = f"wget {link} -O {out_path}/{name}.tar.gz"
    subprocess.call(cmd, shell=True)
    cmd2 = f"tar xzvf {out_path}/{name}.tar.gz -C {out_path}"
    subprocess.call(cmd2, shell=True)