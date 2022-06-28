import torch
import numpy as np
import dgl
from torch import nn
from .se3_dynamics.models import SE3Transformer_dgl, TFN_dgl

class TFN(nn.Module):
    def __init__(self, nf=16, n_layers=3, act_fn=nn.ReLU(), num_degrees=4, div=1):
        super().__init__()
        self.dgl_model = TFN_dgl(num_layers=n_layers, num_channels=nf, edge_dim=0, div=1, act_fn=act_fn, num_degrees=num_degrees)
        self.individual_graph = None

    def forward(self, data):
        if self.individual_graph is None:
            self.individual_graph, self.shape, (self.indices_src, self.indices_dst) = pyg2dgl(data)


        N, D = self.shape
        B = data.num_graphs

        xs = data.pos.reshape(B,N,D)
        vs = data.v.reshape(B,N,D)
        charges = data.z.unsqueeze(1).float()

        distance = xs[:, self.indices_dst] - xs[:, self.indices_src]


        graph = dgl.batch([self.individual_graph] * B).to(data.pos.device)

        graph.ndata['vel'] = vs.view(xs.size(0) * vs.size(1), 3).unsqueeze(1)

        graph.ndata['f1'] = graph.ndata['vel']#torch.cat([self.graph.ndata['x'].unsqueeze(1), self.graph.ndata['vel']], dim=1)


        graph.ndata['f'] = charges.unsqueeze(2)
        graph.edata['d'] = distance.view(-1, 3)


        # 2. Transform G with se3t to G_out
        G_out = self.dgl_model(graph)

        # 3. Transform G_out to out

        out = G_out['1'].view(xs.size())

        # out = xs # TODO transform.

        data.x = (out + xs).reshape(-1,D)

        return data

class SE3Transformer(nn.Module):
    def __init__(self, nf=16, n_layers=3, act_fn=nn.ReLU(), num_degrees=4, div=1):
        super().__init__()
        self.dgl_model = SE3Transformer_dgl(num_layers=n_layers, num_channels=nf, edge_dim=0, div=1, act_fn=act_fn, num_degrees=num_degrees)
        self.individual_graph = None

    def forward(self, data):
        if self.individual_graph is None:
            self.individual_graph, self.shape, (self.indices_src, self.indices_dst) = pyg2dgl(data)


        N, D = self.shape
        B = data.num_graphs

        xs = data.pos.reshape(B,N,D)
        vs = data.v.reshape(B,N,D)
        charges = data.z.unsqueeze(1).float()

        distance = xs[:, self.indices_dst] - xs[:, self.indices_src]


        graph = dgl.batch([self.individual_graph] * B).to(data.pos.device)

        graph.ndata['vel'] = vs.view(xs.size(0) * vs.size(1), 3).unsqueeze(1)

        graph.ndata['f1'] = graph.ndata['vel']#torch.cat([self.graph.ndata['x'].unsqueeze(1), self.graph.ndata['vel']], dim=1)


        graph.ndata['f'] = charges.unsqueeze(2)
        graph.edata['d'] = distance.view(-1, 3)


        # 2. Transform G with se3t to G_out
        G_out = self.dgl_model(graph)

        # 3. Transform G_out to out

        out = G_out['1'].view(xs.size())

        # out = xs # TODO transform.

        data.x = (out + xs).reshape(-1,D)

        return data

def pyg2dgl(data):
    B = data.num_graphs
    N = data.num_nodes // data.num_graphs
    D = data.pos.shape[-1]

    # get neighbour indices here and use throughout entire network; this is a numpy function
    indices_src, indices_dst, _w = connect_fully(N) # [N, K]

    # example has shape [N, D=3]

    # Create graph (connections only, no bond or feature information yet)
    G = dgl.DGLGraph((indices_src, indices_dst))

    example = torch.zeros((N,D))

    G.ndata['x'] = example
    G.ndata['f'] = torch.ones(size=[N, 1, 1])
    G.edata['d'] = example[indices_dst] - example[indices_src] # relative postion

    # individual_graphs = [G] * B


    # batched_graph = dgl.batch(individual_graphs)

    return G, (N,D), (indices_src, indices_dst)

def connect_fully(num_atoms):
    """Convert to a fully connected graph"""
    adjacency = {}
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                adjacency[(i, j)] = 1

    src = []
    dst = []
    w = []
    for edge, weight in adjacency.items():
        src.append(edge[0])
        dst.append(edge[1])
        w.append(weight)

    return np.array(src), np.array(dst), np.array(w)