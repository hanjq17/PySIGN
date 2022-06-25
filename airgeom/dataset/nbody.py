import torch
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
from .utils_physics import System
from tqdm import tqdm
import pickle as pkl
from joblib import Parallel, delayed


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


class NBody(InMemoryDataset):
    raw_url = None

    def __init__(self, root, transform=None, pre_transform=None, n_particle=5, num_samples=500,
                 box_size=None, T=5000, sample_freq=100, num_workers=20,
                 initial_step=15, pred_step=1):
        self.n_particle = n_particle
        self.num_samples = num_samples
        self.box_size = box_size
        self.T, self.sample_freq = T, sample_freq
        self.num_workers = num_workers
        self.initial_step, self.pred_step = initial_step, pred_step
        super(NBody, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def len(self):
        return len(self.slices[list(self.slices.keys())[0]]) - 1

    def get(self, idx):
        data = super(NBody, self).get(idx)
        data.pos = data.pos[self.initial_step]  # the input of x
        data.pred = data.v[self.initial_step + self.pred_step]  # the label of v
        data.v = data.v[self.initial_step]  # the input of v
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
            delayed(para_comp)(self.n_particle, self.box_size, self.T, self.sample_freq) for i in tqdm(range(self.num_samples))
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
            samples.append(Data(pos=cur_loc, v=cur_vel, charge=cur_charge))
        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    dataset = NBody(root='/data/new/cached_datasets/nbody', transform=None)
    print(dataset[10])
