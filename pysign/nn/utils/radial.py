from math import pi

import torch
import torch.nn as nn

__all__ = ['rbf_class_mapping', 'GaussianRBF', 'ExpNormalRBF', 'BesselRBF']


class GaussianRBF(nn.Module):
    r"""Gaussian radial basis functions.

    .. math::

        \phi_i(d) = \exp(\text{coeff} * (d-\text{offset}_i)^2)
    
    where

    .. math::

        \begin{cases}
            \text{offset}_i & = & L + \frac{U-L}{N}i \\
            \text{coeff} & = & -\frac{1}{2}(\frac{U-L}{N})^2
        \end{cases}

    :param cutoff_lower: Lower bound :math:`L` of the cutoff interval.
    :param cutoff_upper: Upper bound :math:`U` of the cutoff interval.
    :param num_rbf: Number of radial basis functions :math:`N`.
    :param trainable: Whether the RBF parameters, i.e. offset and coeff, are trainable.
    """

    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=False):
        super(GaussianRBF, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(self.cutoff_lower, self.cutoff_upper, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        r"""Forward function.

        :param dist: Distance :math:`d`, (num_nodes,)
        :return: :math:`\phi(d)`, (num_nodes, num_rbf)
        """
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ExpNormalRBF(nn.Module):
    r"""ExpNormal radial basis functions proposed in PhysNet.

    .. math::

        \phi_i(d) = \exp(-\beta_i(\exp(\alpha(L-d))-\mu_i)^2)
    
    where

    .. math::

        \begin{cases}
            \mu_i & = s + \frac{1-s}{N}i \\
            \beta_i & = (\frac{2(1-s)}{N})^{-2}N \\
            \alpha & =\frac{5}{U-L} \\
            s & = \exp(L-U)
        \end{cases}

    :param cutoff_lower: Lower bound :math:`L` of the cutoff interval.
    :param cutoff_upper: Upper bound :math:`U` of the cutoff interval.
    :param num_rbf: Number of radial basis functions :math:`N`.
    :param trainable: Whether the RBF parameters, i.e. :math:`\mu` and :math:`\beta`, are trainable.
    """

    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(ExpNormalRBF, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        r"""Forward function.

        :param dist: Distance :math:`d`, (num_nodes,)
        :return: :math:`\phi(d)`, (num_nodes, num_rbf)
        """
        dist = dist.unsqueeze(-1)
        return torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )


class BesselRBF(nn.Module):
    r"""
    Sine for radial basis functions with coulomb decay (0th order bessel).

    References:

    .. Klicpera, Groß, Günnemann:
       Directional message passing for molecular graphs.
       ICLR 2020

    .. math::

        \phi_i(d) = \frac{\sin(\alpha_i(d-L))}{d-L}

    where

    .. math::

        \alpha_i = \frac{\pi i}{U-L}
    
    :param cutoff_lower: Lower bound :math:`L` of the cutoff interval.
    :param cutoff_upper: Upper bound :math:`U` of the cutoff interval.
    :param num_rbf: Number of radial basis functions :math:`N`.
    :param trainable: Whether the RBF parameter :math:`\alpha` is trainable.
    """

    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=False):
        super(BesselRBF, self).__init__()
        self.num_rbf = num_rbf
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.trainable = trainable

        freqs = self._initial_params()
        if self.trainable:
            self.register_parameter("freqs", nn.Parameters(freqs))
        else:
            self.register_buffer("freqs", freqs)

    def _initial_params(self):
        freqs = torch.arange(1, self.num_rbf + 1) * pi / (self.cutoff_upper - self.cutoff_lower)
        return freqs

    def reset_parameters(self):
        freqs = self._initial_params()
        self.freqs.data.copy_(freqs)

    def forward(self, dist):
        r"""Forward function.

        :param dist: Distance :math:`d`, (num_nodes,)
        :return: :math:`\phi(d)`, (num_nodes, num_rbf)
        """
        inputs = dist - self.cutoff_lower
        ax = inputs[..., None] * self.freqs
        sinax = torch.sin(ax)
        norm = torch.where(inputs == 0, torch.tensor(1.0, device=dist.device), inputs)
        y = sinax / norm[..., None]
        return y


rbf_class_mapping = {
    'gaussian': GaussianRBF,
    'expnorm': ExpNormalRBF,
    "bessel": BesselRBF
}
