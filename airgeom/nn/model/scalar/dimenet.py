#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Callable, Optional
from math import pi as PI
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import SparseTensor
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing, radius_graph

from airgeom.nn.model.scalar.schnet import InteractionBlock


def swish(x):
    """
    Swish activation function from Ramachandran, Zopf, Le 2017. "Searching for Activation Functions"
    """
    return x * torch.sigmoid(x)


def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


class DimeNet(nn.Module):
    """
    The directional message passing neural network (DimeNet) by Klicpera et al. 2020

    :param in_node_nf (int, required): Number of features for 'h' in the input
    :param out_node_nf (int, required): Number of features for 'h' in the output
    :params hidden_nf (int, optional): Hidden feature size (default 128)
    :params num_blocks (int, optional): Number of building blocks (default 6)
    :params num_bilinear (int, optional): Size of the bilinear layer tensor (default 8)
    :params num_spherical (int, optional): Number of spherical harmonics (default 7)
    :params num_radial (int, optional): Number of radial basis functions (defualt 6)
    :params cutoff (float, optional): Cutoff distance for interatomic interactions (default 5.0)
    :params max_num_neighbors (int, optional): The maximum number of neighbors to collect for
        each node within the cutoff distance (default 32)
    :parmas envelope_exponent (int, optional): Shape of the smooth cutoff (default 5)
    :params num_before_skip (int, optional): Number of residual layers in the interaciton blocks
        before the skip connection (default 1)
    :params num_after_skip (int, optional): Number of residual layers in the interaction blocks
        after the skipt connection (default 3)
    :params num_output_layers (int, optional): Number of linear layers for the output blocks (default 3)
    :params act (Callable, optional): The activation function (default swish)
    """
    def __init__(self, in_node_nf: int, out_node_nf: int, hidden_nf: int=128,
                 num_blocks: int=6, num_bilinear: int=8, num_spherical: int=7,
                 num_radial: int=6, cutoff: float=5.0, max_num_neighbors: int=32,
                 envelope_exponent: int=5, num_before_skip: int=1,
                 num_after_skip: int=2, num_output_layers: int=3,
                 act: Callable=swish):
        super().__init__()
        
        self.in_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

        self.embed = EmbeddingBlock(in_node_nf, num_radial, hidden_nf, act)

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff, envelope_exponent)

        self.interaction_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.interaction_blocks.append(
                InteractionBlock(hidden_nf, num_bilinear, num_spherical,
                                 num_radial, num_before_skip, num_after_skip, act)
            )

        self.output_blocks = nn.ModuleList()
        for _ in range(num_blocks + 1):
            self.output_blocks.append(
                OutputBlock(num_radial, hidden_nf, out_node_nf,
                num_output_layers, act)
            )

        self.reset_parameters()

    @property
    def params(self):
        """
        Get the parameters to optimize.

        :return: The parameters to optimize.
        """
        return self.parameters()
    
    def reset_parameters(self):
        self.rbf.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    def forward(self, data):
        h, pos, batch = data.h, data.x, data.batch
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        # edge_index = data.edge_index
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index, num_nodes=h.size(0))
        dist = (pos[i] - pos[j]).norm(dim=-1)  # calculate distances

        # calculate angles
        pos_i = pos[idx_i]
        pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # embedding block
        h = self.embed(h, rbf, i, j)
        P = self.output_blocks[0](h, rbf, i, num_nodes=pos.size(0))

        # interaction block
        for interaction_block, output_block in zip(self.interaction_blocks, self.output_blocks[1:]):
            h = interaction_block(h, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(h, rbf, i, num_nodes=data.num_nodes)

        data.x_pred, data.h_pred = pos, P
        return data

    def triplets(self, edge_index, num_nodes):
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        
        adj_t = SparseTensor(row=col, col=row, value=value,
                             sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


class Envelope(torch.nn.Module):
    def __init__(self, exponent):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class BesselBasisLayer(nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.Tensor(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class SphericalBasisLayer(nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        super().__init__()
        import sympy as sym

        from torch_geometric.nn.models.dimenet_utils import (
            bessel_basis,
            real_sph_harm,
        )

        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, in_node_nf, num_radial, hidden_channels, act=swish):
        super().__init__()
        self.act = act

        self.emb = nn.Linear(in_node_nf, hidden_channels)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels)
        self.lin = nn.Linear(3 * hidden_channels, hidden_channels)

    def forward(self, x, rbf, i, j):
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))


class OutputBlock(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, out_channels, num_layers,
                 act=swish):
        super().__init__()
        self.act = act

        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lin = nn.Linear(hidden_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        self.lin.weight.data.fill_(0)

    def forward(self, x, rbf, i, num_nodes=None):
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super().__init__()
        self.act = act
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_bilinear, num_spherical,
                 num_radial, num_before_skip, num_after_skip, act=swish):
        super().__init__()
        self.act = act

        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)
        self.lin_sbf = nn.Linear(num_spherical * num_radial, num_bilinear,
                                 bias=False)

        # Dense transformations of input messages.
        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.W = torch.nn.Parameter(
            torch.Tensor(hidden_channels, num_bilinear, hidden_channels))

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)
        self.W.data.normal_(mean=0, std=2 / self.W.size(0))
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, x, rbf, sbf, idx_kj, idx_ji):
        rbf = self.lin_rbf(rbf)
        sbf = self.lin_sbf(sbf)

        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))
        x_kj = x_kj * rbf
        x_kj = torch.einsum('wj,wl,ijl->wi', sbf, x_kj[idx_kj], self.W)
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h
