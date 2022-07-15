import torch.nn as nn
from ...layer import EGNNLayer
from ..registry import EncoderRegistry


@EncoderRegistry.register_encoder('EGNN')
class EGNN(nn.Module):
    """
    E(n)-equivariant Graph Neural Network by Satorras et al., 2021

    :param in_node_dim: Number of features for 'h' at the input
    :param hidden_dim: Number of hidden features
    :param out_node_dim: Number of features for 'h' at the output
    :param in_edge_dim: Number of features for the edge features
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
    :param use_vel: Whether using velocities as inputs in dynamic simulation.
    """
    def __init__(self, in_node_dim, hidden_dim, out_node_dim=None, in_edge_dim=0, act_fn=nn.SiLU(),
                 n_layers=4, residual=True, attention=False, normalize=False, tanh=False, use_vel=False, **kwargs):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_dim
        self.n_layers = n_layers
        self.use_vel = use_vel
        self.embedding_in = nn.Linear(in_node_dim, self.hidden_nf)
        if out_node_dim is None:
            out_node_dim = hidden_dim
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_dim)
        for i in range(n_layers):
            self.add_module("gcl_%d" % i,
                            EGNNLayer(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_dim,
                                      act_fn=act_fn, residual=residual, attention=attention, use_vel=use_vel,
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
        h, x = data.h, data.x
        edges = data.edge_index
        edge_attr = data.edge_attr
        if hasattr(data, 'v'):
            vel = data.v
        else:
            vel = None
        h = self.embedding_in(h)
        for i in range(self.n_layers):
            if self.use_vel:
                h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, vel=vel)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, vel=vel)
        h = self.embedding_out(h)
        data.x_pred, data.h_pred = x, h
        return data
