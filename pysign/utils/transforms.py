from torch_geometric.utils import remove_self_loops, dense_to_sparse
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T


class ToFullyConnected(object):

    def __init__(self, preserve_edge_attr=True):
        self.preserve_edge_attr = preserve_edge_attr

    def __call__(self, data):
        # device = data.edge_index.device
        row = torch.arange(data.num_nodes, dtype=torch.long)
        col = torch.arange(data.num_nodes, dtype=torch.long)
        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)
        edge_attr = None
        if data.edge_attr is not None and self.preserve_edge_attr:
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

    def __init__(self, max_atom_type=100, charge_power=2, atom_type_name='atom_type', atom_list=None):
        self.max_atom_type = max_atom_type
        self.charge_power = charge_power
        self.atom_type_name = atom_type_name
        self.atom_list = atom_list
        self.atom_map = self.rev_map(atom_list)

    def rev_map(self, atom_list):
        res = {}
        for i, atom in enumerate(atom_list):
            res[atom] = i
        return res

    def __call__(self,data):
        # atom_type = data.atom_type
        assert hasattr(data, self.atom_type_name)
        atom_type = getattr(data, self.atom_type_name)
        if self.charge_power == -1:
            data.h = atom_type
        else:
            if self.atom_map is not None:
                atom_type = torch.tensor([self.atom_map[i.item()] for i in atom_type])
                one_hot_size = len(self.atom_list)
            else:
                one_hot_size = self.max_atom_type
            one_hot = F.one_hot(atom_type, one_hot_size)
            charge_tensor = (atom_type.unsqueeze(-1) / self.max_atom_type).pow(
                torch.arange(self.charge_power + 1., dtype=torch.float32))
            charge_tensor = charge_tensor.view(atom_type.shape + (1, self.charge_power + 1))
            atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(atom_type.shape + (-1,))
            data.h = atom_scalars
        return data


class QM9_Transform(object):
    def __init__(self, charge_power):
        self.transform = T.Compose([ToFullyConnected(preserve_edge_attr=False),
                                    AtomOnehot(max_atom_type=10, charge_power=charge_power,
                                               atom_type_name='charge', atom_list=[1, 6, 7, 8, 9])])

    def __call__(self, data):
        return self.transform(data)


class MD17_Transform(object):
    def __init__(self, max_atom_type, charge_power, atom_type_name, max_hop, cutoff):
        self.max_atom_type = max_atom_type
        self.charge_power = charge_power
        self.atom_type_name = atom_type_name
        self.max_hop = max_hop
        self.cutoff = cutoff
        self.processed = False

    def binarize(self, x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(self, adj, order):

        adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device),
                    self.binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]
        for i in range(2, order + 1):
            adj_mats.append(self.binarize(adj_mats[i - 1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)
        for i in range(1, order + 1):
            order_mat += (adj_mats[i] - adj_mats[i - 1]) * i
        return order_mat

    def gen_fully_connected_with_hop(self, pos):
        nodes = pos.shape[0]
        adj = torch.norm(pos.unsqueeze(0) - pos.unsqueeze(1), p=2, dim=-1)  # n * n
        adj = (adj <= self.cutoff) & (~torch.eye(nodes).bool())
        adj_order = self.get_higher_order_adj_matrix(adj.long(), self.max_hop)
        # adj_order = type_mat
        fc = 1 - torch.eye(pos.shape[0], dtype=torch.long)
        ans = adj_order + fc
        edge_index, edge_type = dense_to_sparse(ans)
        return edge_index, edge_type - 1

    def gen_atom_onehot(self, atom_type):
        if self.charge_power == -1:
            return atom_type
        else:
            one_hot = F.one_hot(atom_type, self.max_atom_type)
            charge_tensor = (atom_type.unsqueeze(-1) / self.max_atom_type).pow(
                torch.arange(self.charge_power + 1., dtype=torch.float32))
            charge_tensor = charge_tensor.view(atom_type.shape + (1, self.charge_power + 1))
            atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(atom_type.shape + (-1,))
            return atom_scalars

    def get_example(self, data):
        assert hasattr(data, self.atom_type_name)
        atom_type = getattr(data, self.atom_type_name)
        self.h = self.gen_atom_onehot(atom_type)
        self.edge_index, self.edge_type = self.gen_fully_connected_with_hop(data.x)
        self.processed = True

    def __call__(self, data):
        assert self.processed
        data.h = self.h
        data.edge_index, data.edge_type = self.edge_index, self.edge_type
        return data


class _NBody_Transform(object):
    def __call__(self, data):
        data.edge_attr = data.charge[data.edge_index[0]] * data.charge[data.edge_index[1]]
        data.h = torch.norm(data.v, dim=-1, keepdim=True)
        return data


NBody_Transform = T.Compose([ToFullyConnected(), _NBody_Transform()])


class SelectEdges(object):
    def __init__(self, edges_between=True):
        self.edges_between = edges_between

    def __call__(self, data):
        data.edge_attr = data.edge_attr.reshape(-1,1)
        if not self.edges_between:
            row, col = data.edge_index
            ins = data.instance
            mask = ins[row] == ins[col]
            data.edge_index = data.edge_index[:, mask]
            data.edge_attr = data.edge_attr[mask]
        return data


class LEP_Transform(object):
    def __call__(self, data):
        data1, data2 = data
        data1.edge_attr = data1.edge_attr.reshape(-1,1)
        data2.edge_attr = data2.edge_attr.reshape(-1,1)
        if isinstance(data1.y, str):
            data1.y = torch.FloatTensor([data1.y == 'A'])
        return data1, data2
