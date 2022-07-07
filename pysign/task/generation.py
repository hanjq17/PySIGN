from torch.nn import Linear, MSELoss, Module, L1Loss
from torch_geometric.nn import global_add_pool
from .basic import BasicTask
from .utils.output_modules import *
from ..nn.utils import SinusoidalPosEmb
from torch.autograd import grad
import torch.nn as nn


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class ConformationGeneration(BasicTask):
    """
    The conformation generation task.

    :param rep: the representation module.
    :param rep_dim: the dimension of the representation.
    """
    def __init__(self, rep, rep_dim, num_steps=1000, decoder_type='DifferentialVector', loss='MAE'):
        super(ConformationGeneration, self).__init__(rep)
        self.rep_dim = rep_dim
        self.task_type = 'Generation'
        assert decoder_type in ['DifferentialVector']
        self.decoder_type = decoder_type
        self.decoder = self.get_decoder()
        self.loss = self.get_loss(loss)
        self.num_steps = num_steps

        sinu_pos_emb = SinusoidalPosEmb(rep_dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(rep_dim, rep_dim),
            nn.GELU(),
            nn.Linear(rep_dim, rep_dim)
        )

        self.betas = linear_beta_schedule(self.num_steps)
        alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def get_decoder(self):
        """
        Instantiate the decoder module. Currently adopt a Linear layer.

        :return: The decoder module.
        """
        if self.decoder_type == 'DifferentialVector':
            decoder = DifferentialVector()
        else:
            raise NotImplementedError('Unsupported decoder type:', self.decoder_type)
        return decoder

    def get_loss(self, loss):
        """
        Instantiate the loss module. Currenly adopt the MSE loss.

        :return: The loss module.
        """
        if loss == 'MAE':
            loss = L1Loss(reduction='none')
        elif loss == 'MSE':
            loss = MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unknown loss type', loss)
        return loss

    @property
    def params(self):
        """
        Get the parameters to optimize.

        :return: The parameters to optimize.
        """
        return self.parameters()

    def forward(self, data):
        """
        Forward passing with the data object. First, the data is processed by the representation module.
        Afterwards, the representation is delivered to the decoder, and the output together with labels yield the loss.

        :param data: The data object.
        :return: The loss computed.
        """
        h, x = data.h.clone(), data.x.clone()

        # add time step embedding
        t = torch.randint(0, self.num_steps, size=(data.num_graphs,), device=h.device).long()
        t = t.index_select(0, data.batch)  # [N]
        t_emb = self.time_mlp(t)  # [N, rep_dim]
        h = h + t_emb

        # add noise
        noise = torch.randn_like(x)
        x = self.sqrt_alphas_cumprod.index_select(0, t) * x + \
            self.sqrt_one_minus_alphas_cumprod.index_select(0, t) * noise
        data.v_label = noise

        data.h, data.x = h, x
        rep = self.rep(data)
        output = self.decoder(rep)  # node-wise rep
        loss = self.loss(output, data.v_label).sum(dim=-1)
        return output, loss, data.v_label


