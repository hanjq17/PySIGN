from torch_geometric.nn import global_mean_pool, global_add_pool
import torch
from torch import nn
from torch.autograd import grad


__all__ = ['GeneralPurposeDecoder']

act_class_mapping = {
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, xv):
        x, v = xv
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v


class GeneralPurposeDecoder(nn.Module):
    def __init__(self, hidden_dim, activation='silu', output_dim=1, decoding='MLP', vector_method='diff',
                 normalize=None, dynamics=False):
        super(GeneralPurposeDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.output_dim = output_dim
        self.decoding = decoding
        self.vector_method = vector_method
        self.normalize = normalize
        self.dynamics = dynamics
        if self.dynamics:
            self.output_dim = self.output_dim + 1  # Reserve an additional channel for dt
        act_class = act_class_mapping[activation]

        if self.decoding == 'MLP':
            self.decoder = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                act_class(),
                nn.Linear(self.hidden_dim // 2, output_dim)
            )

        elif self.decoding == 'GatedBlock':
            self.decoder = nn.Sequential(
                GatedEquivariantBlock(self.hidden_dim, self.hidden_dim // 2, self.hidden_dim // 2,
                                      activation=activation, scalar_activation=True),
                GatedEquivariantBlock(self.hidden_dim // 2, 1, activation=activation)
            )
        else:
            self.decoder = None

    def reset_parameters(self):
        if self.decoding == 'MLP':
            nn.init.xavier_uniform_(self.decoder[0].weight)
            self.decoder[0].bias.data.fill_(0)
            nn.init.xavier_uniform_(self.decoder[2].weight)
            self.decoder[2].bias.data.fill_(0)
        elif self.decoding == 'GatedBlock':
            for layer in self.decoder:
                layer.reset_parameters()
        else:
            pass

    def forward(self, data):
        h = data.h_pred
        v = None

        if self.decoding == 'MLP':
            h = self.decoder(h)
        elif self.decoding == 'GatedBlock':
            v = data.vec
            h, v = self.decoder((h, v))
            v = v.squeeze(-1)
        else:
            pass

        h, dt = (h[..., :-1], h[..., -1]) if self.dynamics else (h, None)

        if self.vector_method == 'diff':
            v = data.x_pred - data.x
        elif self.vector_method == 'gradient':
            output = global_add_pool(h.sum(dim=-1), data.batch)
            if self.normalize is not None:
                output = output * self.normalize[1] + self.normalize[0]  # h * std + mean
            grad_outputs = [torch.ones_like(output)]
            v = - grad(
                [output],
                [data.x],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
        else:
            pass

        if self.dynamics:
            v = v * dt.unsqueeze(-1)

        return h, v

