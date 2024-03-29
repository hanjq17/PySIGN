import torch
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
from .utils_physics import System
from tqdm import tqdm
import pickle as pkl
from joblib import Parallel, delayed
from .registry import DatasetRegistry

__all__ = ['NBody']


def para_comp(n_particle, box_size, T, sample_freq):
    while True:
        X, V = [], []
        system = System(n_isolated=n_particle, n_stick=0, n_hinge=0,
                        box_size=box_size)
        for t in range(T):
            system.simulate_one_step()
            if t % sample_freq == 0:
                X.append(system.X.copy())
                V.append(system.V.copy())
        system.check()
        assert system.is_valid()  # currently do not apply constraint
        if system.is_valid():
            cfg = system.configuration()
            X = np.array(X)
            V = np.array(V)
            # return cfg, X, V, system.edges, system.charges
            return X, V, system.charges


@DatasetRegistry.register_dataset('nbody_dynamics')
class NBody(InMemoryDataset):
    raw_url = None  # Use data generation instead of downloading

    def __init__(self, root, transform=None, pre_transform=None, n_particle=5, num_samples=500,
                 box_size=None, T=5000, sample_freq=100, num_workers=20,
                 initial_step=15, pred_step=1, use_dataset_vel=False):
        self.n_particle = n_particle
        self.num_samples = num_samples
        self.box_size = box_size
        self.T, self.sample_freq = T, sample_freq
        self.num_workers = num_workers
        self.initial_step, self.pred_step = initial_step, pred_step
        self.use_dataset_vel = use_dataset_vel
        assert self.initial_step - self.pred_step >= 0  # since v_input is a differential of positions
        super(NBody, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.mode = 'one_step'  # or 'rollout'
        self.rollout_step = None

    def len(self):
        return len(self.slices[list(self.slices.keys())[0]]) - 1

    def get(self, idx):
        if self.mode == 'one_step':
            data = super(NBody, self).get(idx)
            data.v_label = data.x[self.initial_step + self.pred_step] - data.x[self.initial_step]  # the label of v
            if not self.use_dataset_vel:
                data.v = data.x[self.initial_step] - data.x[self.initial_step - self.pred_step]  # the input of v
            else:
                data.v = data.v[self.initial_step]
            data.x = data.x[self.initial_step]  # the input of x, [N, 3]
        elif self.mode == 'rollout':
            assert not self.use_dataset_vel  # Currently we do not support (x, v) rollout
            data = super(NBody, self).get(idx)
            t_idx = torch.arange(self.initial_step, data.x.shape[0] - self.pred_step, self.pred_step)
            assert len(t_idx) >= self.rollout_step
            t_idx = t_idx[:self.rollout_step]
            data.v_label = data.x[t_idx + self.pred_step] - data.x[t_idx]
            data.v_label = data.v_label.transpose(0, 1)
            data.v = data.x[t_idx] - data.x[t_idx - self.pred_step]
            data.v = data.v.transpose(0, 1)  # [N, T, 3]
            data.x = data.x[t_idx]
            data.x = data.x.transpose(0, 1)
        else:
            raise RuntimeError('Unknown mode for NBody dynamics', self.mode)

        return data

    @property
    def raw_file_names(self):
        return [f'NBody_{self.n_particle}_{self.num_samples}_{self.T}_{self.sample_freq}.pkl']

    @property
    def processed_file_names(self):
        return [f'NBody_{self.n_particle}_{self.num_samples}_{self.T}_{self.sample_freq}.pt']

    def download(self):
        """
        Instead of directly downloading data, run the data generation for NBody dataset.
        :return:
        """
        results = Parallel(n_jobs=self.num_workers)(
            delayed(para_comp)(self.n_particle, self.box_size, self.T, self.sample_freq) for i in
            tqdm(range(self.num_samples))
        )
        loc_all, vel_all, charges_all = zip(*results)  # TODO: Do we really need to save the edges?
        with open(self.raw_paths[0], 'wb') as f:
            pkl.dump((loc_all, vel_all, charges_all), f)
            print('Raw file dumped to', self.raw_paths[0])

    def process(self):
        with open(self.raw_paths[0], 'rb') as f:
            loc_all, vel_all, charges_all = pkl.load(f)

        samples = []
        for loc, vel, charges in zip(loc_all, vel_all, charges_all):
            cur_loc, cur_vel = torch.from_numpy(loc).float(), torch.from_numpy(vel).float()
            cur_charge = torch.from_numpy(charges).float()
            samples.append(Data(x=cur_loc, v=cur_vel, charge=cur_charge))
        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])

    def get_split_by_num(self, n_train, n_val, n_test):
        train_dataset = self[:n_train]
        val_dataset = self[n_train: n_train + n_val]
        test_dataset = self[n_train + n_val: n_train + n_val + n_test]
        return {'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset}

    def default_split(self):
        return self.get_split_by_num(n_train=800, n_val=100, n_test=100)


if __name__ == '__main__':
    dataset = NBody(root='/data/new/cached_datasets/nbody', transform=None)
    print(dataset[10])
