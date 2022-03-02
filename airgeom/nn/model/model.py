import torch.nn as nn
from ..layer import E_GCL


class EGNN(nn.Module):
    """E(n) Equivariant Graph Neural Network

    :param in_node_nf: Number of features for 'h' at the input
    :param hidden_nf: Number of hidden features
    :param out_node_nf: Number of features for 'h' at the output
    :param in_edge_nf: Number of features for the edge features
    :param device: Device (e.g. 'cpu', 'cuda:0',...)
    :param act_fn: Non-linearity
    :param n_layers: Number of layer for the EGNN
    :param residual: Use residual connections, we recommend not changing this one
    :param attention: Whether using attention or not
    :param normalize: Normalizes the coordinates messages such that:
                instead of: :math:`x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)\phi_x(m_ij)`
                we get:     :math:`x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)\phi_x(m_ij)/||x_i - x_j||`
                We noticed it may help in the stability or generalization in some future works.
                We didn't use it in our paper.
    :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                    phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                    We didn't use it in our paper.
    """
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', act_fn=nn.SiLU(),
                 n_layers=4, residual=True, attention=False, normalize=False, tanh=False):


        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
        self.to(self.device)

    @property
    def params(self):
        return self.parameters()

    def forward(self, data):
        h, x = data.x, data.pos  # TODO: change to data.h and data.x after modifying QM9 dataset
        edges = data.edge_index
        edge_attr = data.edge_attr
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)
        # TODO: discuss whether to put x and h back to data
        data.x, data.h = x, h
        return data
