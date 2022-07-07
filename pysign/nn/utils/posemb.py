import math
import torch
from torch import nn

__all__ = ['SinusoidalPosEmb']

class SinusoidalPosEmb(nn.Module):
    r"""
    Sinusoidal postional embedding layer

    .. math::
        PE(pos, 2i) = \sin(pos/10000^{2i/d})
        PE(pos, 2i+1) = \cos(pos/10000^{2i/d})
    
    :param dim: Size of the embedding vector
    """


    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb