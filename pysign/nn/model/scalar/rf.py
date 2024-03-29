import torch.nn as nn
import torch
from ...layer import RadialFieldLayer
from ..registry import EncoderRegistry


@EncoderRegistry.register_encoder('RF')
class RadialField(nn.Module):
    def __init__(self, hidden_dim, in_edge_dim=0, act_fn=nn.SiLU(), n_layers=4, **kwargs):
        """
        Radial Field Layer

        :param hidden_dim: Number of hidden node features.
        :param edge_attr_nf: Number of edge features, default: 0.
        :param act_fn: The activation function, default: nn.SiLU.
        :param n_layers: The number of layers, default: 4.
        """
        super(RadialField, self).__init__()
        self.hidden_nf = hidden_dim
        self.n_layers = n_layers
        for i in range(n_layers):
            self.add_module("gcl_%d" % i,
                            RadialFieldLayer(hidden_nf=hidden_dim, edge_attr_nf=in_edge_dim, act_fn=act_fn))

    @property
    def params(self):
        """
        Get the parameters to optimize.

        :return: The parameters to optimize.
        """
        return self.parameters()

    def forward(self, data):
        """
        Conduct Radial Field message passing on data. Radial Field does not update node feature, and thus
        does not require node feature as input.

        :param data: The data object, including coordinate, edge feature, edge index, etc.
        :return: The updated data object.
        """
        x = data.x
        h = data.h
        edges = data.edge_index
        edge_attr = data.edge_attr
        if hasattr(data, 'v'):
            v = data.v
        else:
            v = torch.ones_like(x)
        vel_norm = torch.sqrt(torch.sum(v ** 2, dim=1).unsqueeze(1))
        for i in range(self.n_layers):
            x, _ = self._modules["gcl_%d" % i](x, vel_norm, v, edges, edge_attr)
        data.x_pred = x
        data.h_pred = h
        return data
