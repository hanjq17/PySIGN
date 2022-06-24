import numpy as np
import os
import torch
from tqdm import tqdm
from atom3d.util.transforms import prot_graph_transform, mol_graph_transform, PairedGraphTransform
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data, Batch, DataLoader, InMemoryDataset
from torch_geometric.data.separate import separate
import atom3d.util.graph as gr
import subprocess
import scipy.spatial as ss
import os.path as osp
import copy
from joblib import Parallel, delayed


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
        return item[self.atom_keys[0]], item[self.atom_keys[1]]


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
        combined_graph = Data(node_feats, edges, edge_feats, y=item['scores']['neglog_aff'], pos=node_pos, instance=node_mask)
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
    # subprocess.call(cmd, shell=True)
    print(cmd)
    cmd2 = f"tar xzvf {out_path}/{name}.tar.gz -C {out_path}"
    print(cmd2)
    # subprocess.call(cmd2, shell=True)

def iter_samples(lmdb, idx):
    return lmdb[idx]

def get_preprocessed_list(lmdb, num_workers):
    res = []
    if num_workers == 1:
        for i,item in enumerate(tqdm(lmdb)):
            res.append(item)
        return res
    res = Parallel(n_jobs=num_workers, backend='threading')(delayed(iter_samples)(lmdb,idx) for idx in tqdm(range(len(lmdb))))
    return res



class Atom3DDataset(InMemoryDataset):
    """Machine learning of accurate energy-conserving molecular force fields (Chmiela et al. 2017)
    This class provides functionality for loading MD trajectories from the original dataset, not the revised versions.
    See http://www.quantum-machine.org/gdml/#datasets for details.
    """

    multi_graph_tasks = ['lep']

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
        elif name == 'ppi':
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
                print(f'specified split {split} not available. Possible values are "sequence-identity-30", "sequence-identity-60".')
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
        """Download an ATOM3D dataset in LMDB format. Available datasets are SMP, PIP, RES, MSP, LBA, LEP, PSR, RSR. Please see `FAQ <datasets target>`_ or `atom3d.ai <atom3d.ai>`_ for more details on each dataset.

        :param name: Three-letter code for dataset (not case-sensitive).
        :type name: str
        :param out_path: Path to directory in which to save downloaded dataset.
        :type out_path: str
        :param split: name of split data to download in LMDB format. Defaults to None, in which case raw (unsplit) dataset is downloaded. Please use :func:`download_split_indices` to get pre-computed split indices for raw datasets.
        :type split: str
        """
        link = self.download_url
        name = self.raw_file_names
        out_path = self.raw_dir
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        cmd = f"wget {link} -O {out_path}/{name}"
        subprocess.call(cmd, shell=True)
        cmd2 = f"tar xzvf {out_path}/{name} -C {out_path}"
        subprocess.call(cmd2, shell=True)        


    def process(self):
        data_dir = self.lmdb_file_dir
        save_dir = self.processed_dir
        if self.task in Atom3DDataset.multi_graph_tasks:
            for split in ['train', 'val', 'test']:
                print(f"Preprocessing {split} data for {self.task} task ...")
                dataset = LMDBDataset(os.path.join(data_dir, split), transform=self.pre_transform)
                samples1, samples2 = [], []
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
                for i, item in enumerate(tqdm(dataset)):
                    samples.append(item)
                # samples = get_preprocessed_list(dataset,self.num_workers)
                data, slices = self.collate(samples)
                torch.save((data, slices), osp.join(save_dir, f'{split}.pt')) 
        print("Preprocess done!")


if __name__ == '__main__':
    # download_dataset('lba', 'cached_datasets/atom3d', 'sequence-identity-30')
    # download_dataset('lep', 'cached_datasets/atom3d', 'protein')
    dataset = Atom3DDataset('cached_datasets/atom3d', 'lep', 'train', dataset_arg='protein')