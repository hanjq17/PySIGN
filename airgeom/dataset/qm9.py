from torch_geometric.datasets import QM9 as QM9_pyg
import torch

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


class QM9(QM9_pyg):

    property_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv', 'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom']

    def __init__(self, root, task, transform=None):
        if isinstance(task, str):
            self.task = task
            assert task in QM9.property_list
            self.target_idx = QM9.property_list.index(task)
        elif isinstance(task, int):
            self.target_idx = task
            self.task = QM9.property_list[self.target_idx]
        super(QM9, self).__init__(root, transform=transform)

    def get(self, idx):
        data = super(QM9, self).get(idx)
        if len(data.y.shape) == 2:
            data.y = data.y[:, self.target_idx]
        return data

    def mean(self):
        y = self.data.y
        return float(y[:, self.target_idx].mean())

    def std(self):
        y = self.data.y
        return float(y[:, self.target_idx].std())

    def atomref(self, max_atom_type):
        if self.target_idx in atomrefs:
            out = torch.zeros(max_atom_type)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[self.target_idx])
            return out.view(-1, 1)
        return None

    def download(self):
        super(QM9, self).download()

    def process(self):
        super(QM9, self).process()

