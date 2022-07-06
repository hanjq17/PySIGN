from torch import nn
import torch
from ...utils import unsorted_segment_sum, unsorted_segment_mean

__all__ = ['EGNNLayer', 'RadialFieldLayer']


class EGNNLayer(nn.Module):
    """
    E(n)-equivariant Message Passing Layer

    :param input_nf: Number of features for 'h' at the input
    :param output_nf: Number of features for 'h' at the output
    :param hidden_nf:  Number of hidden features
    :param edges_in_d: Number of features for the edge features, default: 0
    :param act_fn: Activation function, default: nn.SiLU()
    :param residual: Whether using residual connection or not, default: True
    :param attention: Whether using attention in edge model or not, default: False
    :param normalize: Whether normalizing the coordinates messages , default: False
    :param use_vel: Whether using velocities as inputs in dynamic simulation, default: False
    :param coords_agg: Message aggregation method for coordinates, default: 'mean'
    :param tanh: Whether using tanh at the output of phi_x(m_ij) , default: False
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True,
                 attention=False, normalize=False, use_vel=False, coords_agg='mean', tanh=False):

        super(EGNNLayer, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.use_vel = use_vel
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        if self.use_vel:
            self.coord_mlp_vel = nn.Sequential(
                nn.Linear(input_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, 1))
        

    def edge_model(self, source, target, radial, edge_attr):
        """
        The update of edge message.

        :param source: The embedding of node :math:`i`, :math:`h_i`.
        :param target: The embedding of node :math:`j`, :math:`h_j`.
        :param radial: The redial scalar :math:`||x_i - x_j||^2`.
        :param edge_attr: The edge feature :math:`e_{ij}`.
        :return: The edge message.
        """

        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        """
        The update of node embedding.

        :param x: The input node embedding.
        :param edge_index: The edge index.
        :param edge_attr: The edge feature.
        :param node_attr: The node attribute.
        :return: The aggregated node embedding.
        """
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        """
        The update of coordinate.

        :param coord: The coordinate.
        :param edge_index: The edge index.
        :param coord_diff: The difference in coordinates :math:`x_i - x_j`.
        :param edge_feat: The edge feature.
        :return: The updated coordinates.
        """
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        """
        Compute the radial scalar.

        :param edge_index: The edge index.
        :param coord: The coordinate.
        :return: The radial scalar :math:`||x_i - x_j||^2` and difference in coordinates :math:`x_i - x_j`.
        """
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, vel=None):
        """
        The update of a layer.

        :param h: The node embedding.
        :param edge_index: The edge index.
        :param coord: The coordinate :math:`x`.
        :param edge_attr: The edge feature.
        :param node_attr: The node feature.
        :return: The updated node feature, coordinate and edge feature.
        """
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        if self.use_vel and vel is not None:
            coord = coord + self.coord_mlp_vel(h) * vel
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class RadialFieldLayer(nn.Module):
    """
    Radial Field Layer

    :param hidden_nf: Number of hidden features, default: 64
    :param edge_attr_nf: Number of input edge features, default: 0
    :param act_fn: The activation function, default: LeakyReLU
    """
    def __init__(self, hidden_nf=64, edge_attr_nf=0, act_fn=nn.LeakyReLU(0.2)):
        super(RadialFieldLayer, self).__init__()
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(1, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi = nn.Sequential(nn.Linear(1 + edge_attr_nf, hidden_nf), act_fn, layer, nn.Tanh())

    def forward(self, x, vel_norm, vel, edge_index, edge_attr=None):
        """
        The update of a layer.

        :param x: The input position.
        :param vel_norm: The norm of velocity.
        :param vel: The input velocity.
        :param edge_index: The edge index.
        :param edge_attr: The edge features.
        :return: The updated position.
        """
        row, col = edge_index
        edge_m = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_m)
        x = x + vel * self.coord_mlp_vel(vel_norm)
        return x, edge_attr

    def edge_model(self, source, target, edge_attr):
        """
        The update of edge message.

        :param source: The position of node :math:`i`, :math:`x_i`.
        :param target: The position of node :math:`j`, :math:`x_j`.
        :param edge_attr: The edge feature :math:`e_{ij}`.
        :return: The edge message.
        """
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        if edge_attr is not None:
            e_input = torch.cat([radial, edge_attr], dim=1)
        else:
            e_input = radial
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        return m_ij

    def node_model(self, x, edge_index, edge_m):
        """
        The update of node position.

        :param x: The input node embedding.
        :param edge_index: The edge index.
        :param edge_m: The edge message.
        :return: The aggregated node position.
        """
        row, col = edge_index
        agg = unsorted_segment_mean(edge_m, row, num_segments=x.size(0))
        x_out = x + agg
        return x_out

