from .scalar import *
from .mixing import *
from .irreps import *
from .general_decoder import GeneralPurposeDecoder
from .registry import EncoderRegistry


def get_model_from_args(node_dim, edge_attr_dim, args, dynamics=False):
    encoder = EncoderRegistry.get_encoder(args.name)

    _model_args = {'in_node_dim': node_dim, 'in_edge_dim': edge_attr_dim, 'use_vel': dynamics}
    _model_args.update(args)
    rep_model = encoder(**_model_args)

    return rep_model
