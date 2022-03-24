import math
import torch
from torch import nn

__all__ = ['CosineCutoff']


class CosineCutoff(nn.Module):
    r"""
    Cosine cutoff layer
    .. math::
        cutoff(d) & = \begin{cases}
            \frac{1}{2}(\cos(2\frac{d-L}{U-L}+1)+1), & \text{if } L\leq d\leq U
            0, & \text(others)
        \end{cases}
        

    :param cutoff_lower: Lower bound :math:`L` of the cutoff interval. If cutoff_lower < 0, it will be set to 0.
    :param cutoff_upper: Upper bound :math:`U` of the cutoff interval.
    """
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        r"""Forward Function

        :param distances: Distance vector :math: `d`
        :return: Cutoff result with the same shape as :math: `d`
        """
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs
