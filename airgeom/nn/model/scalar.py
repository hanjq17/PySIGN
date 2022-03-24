import torch.nn as nn
import torch
from ..layer import EGNNLayer, RadialFieldLayer

__all__ = ['EGNN', 'RadialFieldLayer']


class EGNN(nn.Module):
    """
    E(n)-equivariant Graph Neural Network by Satorras et al., 2021

    :param in_node_nf: Number of features for 'h' at the input
    :param hidden_nf: Number of hidden features
    :param out_node_nf: Number of features for 'h' at the output
    :param in_edge_nf: Number of features for the edge features
    :param act_fn: Non-linearity
    :param n_layers: Number of layer
    :param residual: Use residual connections, and we recommend not changing this one
    :param attention: Whether using attention or not
    :param normalize: Normalizes the coordinates messages such that instead of
        :math:`x^{l+1}_i = x^{l}_i + \sum(x_i - x_j)\phi_x(m_{ij})`, we employ
        :math:`x^{l+1}_i = x^{l}_i + \sum(x_i - x_j)\phi_x(m_{ij})/||x_i - x_j||`.
        It may help in the stability or generalization in some future works.
    :param tanh: Sets a tanh activation function at the output of :math:`\phi_x(m_{ij})`.
        It bounds the output of :math:`\phi_x(m_{ij})` which improves in stability but it may decrease in accuracy.
    """
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, act_fn=nn.SiLU(),
                 n_layers=4, residual=True, attention=False, normalize=False, tanh=False):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(n_layers):
            self.add_module("gcl_%d" % i, EGNNLayer(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                    act_fn=act_fn, residual=residual, attention=attention,
                                                    normalize=normalize, tanh=tanh))

    @property
    def params(self):
        """
        Get the parameters to optimize.

        :return: The parameters to optimize.
        """
        return self.parameters()

    def forward(self, data):
        """
        Conduct EGNN message passing on data.

        :param data: The data object, including node feature, coordinate, edge feature, edge index, etc.
        :return: The updated data object.
        """
        h, x = data.x, data.pos  # TODO: change to data.h and data.x after modifying QM9 dataset
        edges = data.edge_index
        edge_attr = data.edge_attr
        h = self.embedding_in(h)
        for i in range(self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)
        # TODO: discuss whether to put x and h back to data
        data.x, data.h = x, h
        return data


class RadialField(nn.Module):
    def __init__(self, hidden_nf, edge_attr_nf=0, act_fn=nn.SiLU(), n_layers=4):
        """
        Radial Field Layer

        :param hidden_nf: Number of hidden node features.
        :param edge_attr_nf: Number of edge features, default: 0.
        :param act_fn: The activation function, default: nn.SiLU.
        :param n_layers: The number of layers, default: 4.
        """
        super(RadialField, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        for i in range(n_layers):
            self.add_module("gcl_%d" % i, RadialFieldLayer(hidden_nf=hidden_nf, edge_attr_nf=edge_attr_nf, act_fn=act_fn))

    def forward(self, data):
        """
        Conduct Radial Field message passing on data. Radial Field does not update node feature, and thus
        does not require node feature as input.

        :param data: The data object, including coordinate, edge feature, edge index, etc.
        :return: The updated data object.
        """
        x = data.pos
        edges = data.edge_index
        edge_attr = data.edge_attr
        if 'v' in data.__dict__:
            v = data.v
        else:
            v = torch.ones_like(x)
        vel_norm = torch.sqrt(torch.sum(v ** 2, dim=1).unsqueeze(1))
        for i in range(self.n_layers):
            x, _ = self._modules["gcl_%d" % i](x, vel_norm, v, edges, edge_attr)
        data.x = x
        return data
