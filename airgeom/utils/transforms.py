from torch_geometric.utils import remove_self_loops
import torch
import torch.nn.functional as F


class ToFullyConnected(object):
    def __call__(self, data):
        # device = data.edge_index.device
        row = torch.arange(data.num_nodes, dtype=torch.long)
        col = torch.arange(data.num_nodes, dtype=torch.long)
        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)
        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index
        return data

class AtomOnehot(object):

    def __init__(self,max_atom_type=100, charge_power=2, atom_type_name='atom_type'):
        self.max_atom_type = max_atom_type
        self.charge_power = charge_power
        self.atom_type_name = atom_type_name

    def __call__(self,data):
        #atom_type = data.atom_type
        assert hasattr(data,self.atom_type_name)
        atom_type = getattr(data, self.atom_type_name)
        if self.charge_power == -1:
            data.x = atom_type
        else:
            one_hot = F.one_hot(atom_type, self.max_atom_type)
            charge_tensor = (atom_type.unsqueeze(-1) / self.max_atom_type).pow(
                torch.arange(self.charge_power + 1., dtype=torch.float32))
            charge_tensor = charge_tensor.view(atom_type.shape + (1, self.charge_power + 1))
            atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(atom_type.shape + (-1,))
            data.x = atom_scalars
        return data
