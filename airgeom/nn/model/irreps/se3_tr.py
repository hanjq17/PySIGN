from torch import nn
from .modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GNormBias
from .fibers import Fiber


class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers: int, atom_feature_size: int,
                 num_channels: int, num_nlayers: int = 1, num_degrees: int = 4,
                 edge_dim: int = 4, div: float = 4, pooling: str = 'avg',
                 n_heads: int = 1, use_vel: bool = False, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.pooling = pooling
        self.n_heads = n_heads
        self.use_vel = use_vel

        self.fibers = {'in': Fiber(dictionary={0: atom_feature_size} if not use_vel else {0: atom_feature_size, 1: 1}),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(dictionary={0: self.num_degrees * self.num_channels, 1: 2})}

        blocks = self._build_gcn(self.fibers, num_channels)
        self.Gblock, self.mapping = blocks

    def _build_gcn(self, fibers, out_dim):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim,
                                  div=self.div, n_heads=self.n_heads))
            # Gblock.append(GNormSE3(fibers['mid']))  # need discussion, why use different normalization

            # Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim, div=self.div, n_heads=self.n_heads,
            #                       learnable_skip=True, skip='cat', selfint=self.si_m, x_ij=self.x_ij))
            Gblock.append(GNormBias(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        mapping = nn.Linear(self.num_degrees * self.num_channels, out_dim)

        return nn.ModuleList(Gblock), mapping

    def forward(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        # Compute equivariant weight basis from relative positions
        dis = data.x[edge_index[0]] - data.x[edge_index[1]]
        basis, r = get_basis_and_r(dis, self.num_degrees-1)

        if self.use_vel:
            assert hasattr(data, 'v')
            node_features = {'0': data.h.unsqueeze(-1), '1': data.v.unsqueeze(-2)}
        else:
            node_features = {'0': data.h.unsqueeze(-1)}
        for layer in self.Gblock:
            node_features = layer(node_features=node_features, edge_index=edge_index, edge_attr=edge_attr,
                                  r=r, basis=basis, x=data.x)
        h_pred = self.mapping(node_features['0'][..., -1])
        x_pred = node_features['1'][:, 0, :]
        data.h_pred = h_pred
        data.x_pred = x_pred

        return data
