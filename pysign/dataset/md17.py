import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
import numpy as np
from tqdm import tqdm
import os
import argparse
from .registry import DatasetRegistry

__all__ = ['MD17', 'MD17_Dynamics']


@DatasetRegistry.register_dataset('md17')
class MD17(InMemoryDataset):
    """Machine learning of accurate energy-conserving molecular force fields (Chmiela et al. 2017)
    This class provides functionality for loading MD trajectories from the original dataset, not the revised versions.
    See http://www.quantum-machine.org/gdml/#datasets for details.
    """

    raw_url = "http://www.quantum-machine.org/gdml/data/npz/"

    molecule_files = dict(
        aspirin="aspirin_dft.npz",
        benzene="benzene_old_dft.npz",
        ethanol="ethanol_dft.npz",
        malonaldehyde="malonaldehyde_dft.npz",
        naphthalene="naphthalene_dft.npz",
        salicylic_acid="salicylic_dft.npz",
        toluene="toluene_dft.npz",
        uracil="uracil_dft.npz",
    )

    available_molecules = list(molecule_files.keys())

    def __init__(self, root, transform=None, pre_transform=None, dataset_arg=None):
        assert dataset_arg is not None, (
            "Please provide the desired comma separated molecule(s) through"
            f"'dataset_arg'. Available molecules are {', '.join(MD17.available_molecules)} "
            "or 'all' to train on the combined dataset."
        )

        if dataset_arg == "all":
            dataset_arg = ",".join(MD17.available_molecules)
        self.molecules = dataset_arg.split(",")

        if len(self.molecules) > 1:
            print(
                "MD17 molecules have different reference energies, "
                "which is not accounted for during training."
            )

        super(MD17, self).__init__(root, transform, pre_transform)

        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

    def len(self):
        return len(self.slices[list(self.slices.keys())[0]]) - 1

    def get(self, idx):
        data = super(MD17, self).get(idx)
        return data

    def mean(self):
        y = self.data.y
        return float(y.mean())

    def std(self):
        y = self.data.y
        return float(y.std())

    @property
    def raw_file_names(self):
        return [MD17.molecule_files[mol] for mol in self.molecules]

    @property
    def processed_file_names(self):
        return [f"md17-{mol}.pt" for mol in self.molecules]

    def download(self):
        for file_name in self.raw_file_names:
            download_url(MD17.raw_url + file_name, self.raw_dir)

    def process(self):
        for path in self.raw_paths:
            data_npz = np.load(path)
            z = torch.from_numpy(data_npz["z"]).long()
            positions = torch.from_numpy(data_npz["R"]).float()
            energies = torch.from_numpy(data_npz["E"]).float()
            forces = torch.from_numpy(data_npz["F"]).float()

            samples = []
            for pos, y, dy in zip(positions, energies, forces):
                samples.append(Data(charge=z, x=pos, y=y.unsqueeze(1), dy=dy))

            if self.pre_filter is not None:
                samples = [data for data in samples if self.pre_filter(data)]

            if self.pre_transform is not None:
                samples = [self.pre_transform(data) for data in tqdm(samples)]

            data, slices = self.collate(samples)
            torch.save((data, slices), self.processed_paths[0])

    def get_split_by_num(self, n_train, n_val, n_test):
        n_tot = len(self)
        import numpy as np
        # Generate random permutation
        np.random.seed(0)
        data_perm = np.random.permutation(n_tot)

        # Now use the permutations to generate the indices of the dataset splits.
        train, valid, test, extra = np.split(
            data_perm, [n_train, n_train + n_val, n_train + n_val + n_test])
        train_dataset, val_dataset, test_dataset = self[train], self[valid], self[test]
        return {'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset}

    def default_split(self):
        n_train = 950
        n_val = 50
        n_test = len(self) - n_train - n_val
        return self.get_split_by_num(n_train=n_train, n_val=n_val, n_test=n_test)


@DatasetRegistry.register_dataset('md17_dynamics')
class MD17_Dynamics(MD17):

    def __init__(self, root, transform=None, pre_transform=None, dataset_arg=None, vel_step=0, pred_step=1):
        super(MD17_Dynamics, self).__init__(root, transform, pre_transform, dataset_arg)
        self.vel_step = vel_step
        self.pred_step = pred_step
        self.mode = 'one_step'  # or 'rollout'
        self.rollout_step = None

    def len(self):
        if self.mode == 'one_step':
            return super(MD17_Dynamics, self).len()
        elif self.mode == 'rollout':
            return super(MD17_Dynamics, self).len() // (self.rollout_step * self.pred_step)
        else:
            raise NotImplementedError()

    def get(self, idx):
        if self.mode == 'one_step':
            prev_idx = idx if idx - self.vel_step < 0 else idx - self.vel_step
            next_idx = idx if idx + self.pred_step >= self.len() else idx + self.pred_step
            data, data_prev, data_next = super(MD17_Dynamics, self).get(idx), super(MD17_Dynamics, self).get(
                prev_idx), super(MD17_Dynamics, self).get(next_idx)
            data.v = (data.x - data_prev.x) / self.vel_step * self.pred_step
            data.v_label = data_next.x - data.x
        elif self.mode == 'rollout':
            raise NotImplementedError()
            # t_idx = torch.arange(start=idx - self.pred_step, end=idx + self.rollout_step * self.pred_step,
            #                      step=self.pred_step).tolist()
            # trajectory = [super(MD17_Dynamics, self).get(_) for _ in t_idx]
            # trajectory = torch.stack(trajectory, dim=1)
        else:
            raise NotImplementedError()

        return data

    def default_split(self):
        n_train = 9500
        n_val = 500
        n_test = 10000
        return self.get_split_by_num(n_train=n_train, n_val=n_val, n_test=n_test)


def get_mean_std(dataloaders):
    val_loader = dataloaders['val']
    ys = torch.cat([batch.y.squeeze() for batch in val_loader])
    return ys.mean(), ys.std()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MD17 preprocess')
    parser.add_argument('--base_path', type=str, default='.', metavar='N',
                        help='Path of download.')
    args = parser.parse_args()

    molecules = ['aspirin', 'benzene', 'ethanol', 'malonaldehyde', 'naphthalene', 'salicylic_acid', 'toluene', 'uracil']

    for mol in molecules:
        base_path = os.path.join(args.base_path, mol)
        os.makedirs(base_path, exist_ok=True)
        dataset = MD17(base_path, dataset_arg=mol)
