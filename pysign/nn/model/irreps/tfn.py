from torch import nn
from .utils.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GNormBias
from .utils.fibers import Fiber
from ..registry import EncoderRegistry


@EncoderRegistry.register_encoder('TFN')
class TFN(nn.Module):
    """SE(3) equivariant GCN"""

    def __init__(self, n_layers: int, in_node_dim: int,
                 hidden_dim: int, num_nlayers: int = 1, num_degrees: int = 4,
                 in_edge_dim: int = 4, use_vel: bool = False, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = n_layers
        self.num_nlayers = num_nlayers
        self.num_channels = hidden_dim
        self.num_degrees = num_degrees
        self.use_vel = use_vel
        self.num_channels_out = hidden_dim * num_degrees
        self.edge_dim = in_edge_dim

        self.fibers = {'in': Fiber(dictionary={0: in_node_dim} if not use_vel else {0: in_node_dim, 1: 1}),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(dictionary={0: self.num_channels_out, 1: 2})}

        self.blocks, self.mapping = self._build_gcn(self.fibers, hidden_dim)

    def _build_gcn(self, fibers, out_dim):

        blocks = []
        fin = fibers['in']
        for i in range(self.num_layers - 1):
            blocks.append(GConvSE3(fin, fibers['mid'], self_interaction=True, edge_dim=self.edge_dim))
            blocks.append(GNormSE3(fibers['mid'], num_layers=self.num_nlayers))
            # blocks.append(GNormBias(fibers['mid']))  # TODO: check
            fin = fibers['mid']
        blocks.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))
        mapping = nn.Linear(self.num_channels_out, out_dim)

        return nn.ModuleList(blocks), mapping

    def forward(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        # Compute equivariant weight basis from relative positions
        dis = data.x[edge_index[0]] - data.x[edge_index[1]]  # TODO: check
        basis, r = get_basis_and_r(dis, self.num_degrees - 1)

        if self.use_vel:
            assert hasattr(data, 'v')
            node_features = {'0': data.h.unsqueeze(-1), '1': data.v.unsqueeze(-2)}
        else:
            node_features = {'0': data.h.unsqueeze(-1)}
        for layer in self.blocks:
            node_features = layer(node_features=node_features, edge_index=edge_index, edge_attr=edge_attr,
                                  r=r, basis=basis)

        h_pred = self.mapping(node_features['0'][..., -1])
        x_pred = node_features['1'][:, 0, :]
        data.h_pred = h_pred
        data.x_pred = x_pred + data.x
        return data
