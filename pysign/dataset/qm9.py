from torch_geometric.datasets import QM9 as QM9_pyg
from pysign.data import from_pyg
import torch
import numpy as np
import torch.nn.functional as F
from .registry import DatasetRegistry

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}


@DatasetRegistry.register_dataset('qm9')
class QM9(QM9_pyg):
    property_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv', 'u0_atom',
                     'u298_atom', 'h298_atom', 'g298_atom']

    def __init__(self, root, task, transform=None, dataset_seed=0, split='large'):
        if isinstance(task, str):
            self.task = task
            assert task in QM9.property_list
            self.target_idx = QM9.property_list.index(task)
        elif isinstance(task, int):
            self.target_idx = task
            self.task = QM9.property_list[self.target_idx]
        attrs = ['x', 'z', 'pos', 'edge_index', 'edge_attr', 'y', 'name', 'idx']
        self.dataset_seed = dataset_seed
        self.split = split
        super(QM9, self).__init__(root, transform=transform, pre_transform=from_pyg(attrs=attrs))
        self.standarize()

    def get(self, idx):
        data = super(QM9, self).get(idx)
        if len(data.y.shape) == 2:
            data.y = data.y[:, self.target_idx]
        if self.target_idx in atomrefs:
            self.substract_thermo(data)
        return data

    def standarize(self):
        ys = torch.tensor([self.get(idx).y for idx in range(self.len())])
        self._mean = ys.mean()
        self._std = ys.std()

    def mean(self):
        return self._mean

    def std(self):
        return self._std

    def atomref(self, max_atom_type):
        if self.target_idx in atomrefs:
            out = torch.zeros(max_atom_type)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[self.target_idx])
            return out.view(-1)
        return None

    def substract_thermo(self, data):
        max_atom_type = 10
        atom_onehot = F.one_hot(data.charge, max_atom_type)
        atom_ref = self.atomref(max_atom_type)
        thermo = (atom_onehot.sum(dim = 0) * atom_ref).sum()
        data.y = data.y - thermo

    def download(self):
        super(QM9, self).download()

    def process(self):
        super(QM9, self).process()

    def get_split_by_num(self, n_train, n_val, n_test):

        print('Getting split with dataset seed (numpy)', self.dataset_seed)
        np.random.seed(self.dataset_seed)
        data_perm = np.random.permutation(len(self))

        # Now use the permutations to generate the indices of the dataset splits.
        train, valid, test, extra = np.split(
            data_perm, [n_train, n_train + n_val, n_train + n_val + n_test])
        train_dataset, val_dataset, test_dataset = self[train], self[valid], self[test]
        return {'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset}

    def default_split(self):

        if self.split == 'large':
            n_train = 110000
            n_val = 10000
            n_test = len(self) - n_train - n_val
        else:
            n_tot = len(self)
            n_test = int(0.1 * n_tot)
            n_train = 100000
            n_val = n_tot - n_train - n_test

        return self.get_split_by_num(n_train=n_train, n_val=n_val, n_test=n_test)
