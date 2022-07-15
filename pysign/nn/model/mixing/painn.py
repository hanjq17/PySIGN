import torch
import torch.nn as nn
from ...layer import PaiNNInteraction, PaiNNMixing, replicate_module
from ...utils import rbf_class_mapping, CosineCutoff
from ..registry import EncoderRegistry

__all__ = ['PaiNN']


@EncoderRegistry.register_encoder('PaiNN')
class PaiNN(nn.Module):
    """PaiNN - polarizable interaction neural network

    References:

    .. Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html

    """

    def __init__(
            self,
            hidden_dim: int,
            n_layers: int,
            radial_basis: nn.Module = None,
            rbf_type: str = "gaussian",
            num_rbf: int = 50,
            trainable_rbf: float = False,
            cutoff_lower: float = 0.0,
            cutoff_upper: float = 5.0,
            activation: nn.Module = nn.SiLU(),
            in_node_dim: int = 100,
            shared_interactions: bool = False,
            shared_filters: bool = False,
            epsilon: float = 1e-8,
            **kwargs
    ):
        """
        Args:
            hidden_dim: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_layers: number of interaction blocks.
            radial_basis: layer for expanding inter-atomic distances in a basis set
            activation: activation function
            shared_interactions: if True, share the weights across
                interaction blocks.
            shared_interactions: if True, share the weights across
                filter-generating networks.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(PaiNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.radial_basis = radial_basis or rbf_class_mapping[rbf_type](cutoff_lower=cutoff_lower,
                                                                        cutoff_upper=cutoff_upper,
                                                                        num_rbf=num_rbf, trainable=trainable_rbf)

        self.cutoff_fn = CosineCutoff(cutoff_lower, cutoff_upper)

        self.embedding = nn.Linear(in_node_dim, hidden_dim)

        self.share_filters = shared_filters

        if shared_filters:
            self.filter_net = nn.Linear(
                self.radial_basis.num_rbf, 3 * hidden_dim
            )
        else:
            self.filter_net = nn.Linear(
                self.radial_basis.num_rbf,
                self.n_layers * hidden_dim * 3
            )

        self.interactions = replicate_module(
            lambda: PaiNNInteraction(
                n_atom_basis=self.hidden_dim, activation=activation
            ),
            self.n_layers,
            shared_interactions,
        )
        self.mixing = replicate_module(
            lambda: PaiNNMixing(
                n_atom_basis=self.hidden_dim, activation=activation, epsilon=epsilon
            ),
            self.n_layers,
            shared_interactions,
        )

    @property
    def params(self):
        """
        Get the parameters to optimize.

        :return: The parameters to optimize.
        """
        return self.parameters()

    def forward(self, data):
        """
        Compute atomic representations/embeddings.

        Args:
            data: The input data object.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        # get tensors from input dictionary
        atomic_numbers = data.h
        x = data.x
        idx_i, idx_j = data.edge_index
        r_ij = x[idx_j] - x[idx_i]
        n_atoms = atomic_numbers.shape[0]

        # compute atom and pair features
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij
        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij) if self.cutoff_fn else torch.ones_like(d_ij)

        filters = self.filter_net(phi_ij) * fcut[..., None]
        if self.share_filters:
            filter_list = [filters] * self.n_layers
        else:
            filter_list = torch.split(filters, 3 * self.hidden_dim, dim=-1)

        q = self.embedding(atomic_numbers)[:, None]
        qs = q.shape

        if hasattr(data, 'v'):
            mu = data.v.unsqueeze(-1).repeat(1, 1, qs[2])
        else:
            mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)

        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)

        q = q.squeeze(1)

        data.x_pred = x
        data.h_pred = q
        data.vec = mu
        return data
