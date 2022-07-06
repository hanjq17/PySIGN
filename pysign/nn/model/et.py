import torch
from torch import nn
from ..layer import NeighborEmbedding, act_class_mapping, EquivariantMultiHeadAttention
from ..utils import CosineCutoff, rbf_class_mapping
from .registry import EncoderRegistry

__all__ = ['EquivariantTransformer']


@EncoderRegistry.register_encoder('ET')
class EquivariantTransformer(nn.Module):
    r"""The TorchMD equivariant Transformer architecture.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_layers (int, optional): The number of attention layers.
            (default: :obj:`6`)
        num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
            (default: :obj:`50`)
        rbf_type (string, optional): The type of radial basis function to use.
            (default: :obj:`"expnorm"`)
        trainable_rbf (bool, optional): Whether to train RBF parameters with
            back-propagation. (default: :obj:`True`)
        activation (string, optional): The type of activation function to use.
            (default: :obj:`"silu"`)
        attn_activation (string, optional): The type of activation function to use
            inside the attention mechanism. (default: :obj:`"silu"`)
        neighbor_embedding (bool, optional): Whether to perform an initial neighbor
            embedding step. (default: :obj:`True`)
        num_heads (int, optional): Number of attention heads.
            (default: :obj:`8`)
        distance_influence (string, optional): Where distance information is used inside
            the attention mechanism. (default: :obj:`"both"`)
        cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
            (default: :obj:`0.0`)
        cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
            (default: :obj:`5.0`)
        max_z (int, optional): Maximum atomic number. Used for initializing embeddings.
            (default: :obj:`100`)
        max_num_neighbors (int, optional): Maximum number of neighbors to return for a
            given node/atom when constructing the molecular graph during forward passes.
            This attribute is passed to the torch_cluster radius_graph routine keyword
            max_num_neighbors, which normally defaults to 32. Users should set this to
            higher values if they are using higher upper distance cutoffs and expect more
            than 32 neighbors per node/atom.
            (default: :obj:`32`)
    """

    def __init__(
        self,
        hidden_channels=128,
        num_layers=6,
        num_rbf=50,
        rbf_type="expnorm",
        trainable_rbf=True,
        activation="silu",
        attn_activation="silu",
        neighbor_embedding=True,
        num_heads=8,
        distance_influence="both",
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_z=100,
        max_num_neighbors=32,
    ):
        super(EquivariantTransformer, self).__init__()

        assert distance_influence in ["keys", "values", "both", "none"]
        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )
        assert attn_activation in act_class_mapping, (
            f'Unknown attention activation function "{attn_activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_z = max_z

        act_class = act_class_mapping[activation]

        self.embedding = nn.Linear(self.max_z, hidden_channels)

        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )

        self.cutoff_fn = CosineCutoff(cutoff_lower, cutoff_upper)

        self.neighbor_embedding = (
            NeighborEmbedding(
                hidden_channels, num_rbf, cutoff_lower, cutoff_upper, self.max_z
            ).jittable()
            if neighbor_embedding
            else None
        )

        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = EquivariantMultiHeadAttention(
                hidden_channels,
                num_rbf,
                distance_influence,
                num_heads,
                act_class,
                attn_activation,
                cutoff_lower,
                cutoff_upper,
            ).jittable()
            self.attention_layers.append(layer)

        self.out_norm = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()

    @property
    def params(self):
        """
        Get the parameters to optimize.

        :return: The parameters to optimize.
        """
        return self.parameters()

    def forward(self, data):
        z = data.h
        x = self.embedding(z)
        pos = data.x
        edge_index = data.edge_index
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        edge_weight = torch.norm(edge_vec, dim=-1)
        
        edge_attr = self.cutoff_fn(edge_weight.unsqueeze(-1)) * self.distance_expansion(edge_weight)
        
        edge_vec = edge_vec / edge_weight.unsqueeze(1)

        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)

        # TODO: need discussions here
        if hasattr(data, 'v'):
            vec = data.v.unsqueeze(-1).repeat(1,1,x.size(1))
        else:
            vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)

        for attn in self.attention_layers:
            dx, dvec = attn(x, vec, edge_index, edge_weight, edge_attr, edge_vec)
            x = x + dx
            vec = vec + dvec
        x = self.out_norm(x)

        data.h_pred = x
        data.x_pred = pos
        data.vec = vec

        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"rbf_type={self.rbf_type}, "
            f"trainable_rbf={self.trainable_rbf}, "
            f"activation={self.activation}, "
            f"attn_activation={self.attn_activation}, "
            f"neighbor_embedding={self.neighbor_embedding}, "
            f"num_heads={self.num_heads}, "
            f"distance_influence={self.distance_influence}, "
            f"cutoff_lower={self.cutoff_lower}, "
            f"cutoff_upper={self.cutoff_upper})"
        )


