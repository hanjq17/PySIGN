from .scalar import *
from .painn import *
from .et import *
from .irreps import TFN, SE3Transformer
from .general_decoder import GeneralPurposeDecoder
from .registry import EncoderRegistry


def get_model_from_args(node_dim, edge_attr_dim, args, dynamics=False):
    encoder = EncoderRegistry.get_encoder(args.name)

    # TODO: change the logic here --- define model specific args
    if args.name == 'EGNN':
        rep_model = encoder(in_node_nf=node_dim, hidden_nf=args.hidden_dim, out_node_nf=args.hidden_dim,
                            in_edge_nf=edge_attr_dim, n_layers=args.n_layers, use_vel=dynamics)
    elif args.name == 'RF':
        rep_model = encoder(hidden_nf=args.hidden_dim, edge_attr_nf=edge_attr_dim, n_layers=args.n_layers)
    elif args.name == 'SchNet':
        rep_model = encoder(in_node_nf=node_dim, out_node_nf=args.hidden_dim, hidden_nf=args.hidden_dim,
                            num_interactions=args.n_layers)
    # elif args.name == 'DimeNet':
    #     rep_model = encoder(in_node_nf=node_dim, out_node_nf=args.hidden_dim, hidden_nf=args.hidden_dim,
    #                         num_blocks=args.n_layers)
    elif args.name == 'PaiNN':
        rep_model = encoder(max_z=node_dim, n_atom_basis=args.hidden_dim, n_interactions=args.n_layers)
    elif args.name == 'ET':
        rep_model = encoder(max_z=node_dim, hidden_channels=args.hidden_dim, num_layers=args.n_layers)
    elif args.name == 'TFN':
        rep_model = encoder(num_layers=args.n_layers, atom_feature_size=node_dim, num_channels=args.hidden_dim,
                            num_nlayers=1, num_degrees=2, edge_dim=edge_attr_dim, use_vel=dynamics)
    elif args.name == 'SE3Transformer':
        rep_model = encoder(num_layers=args.n_layers, atom_feature_size=node_dim,
                            num_channels=args.hidden_dim, num_nlayers=1, num_degrees=2, edge_dim=edge_attr_dim,
                            n_heads=2, use_vel=dynamics)
    else:
        raise NotImplementedError('Not implemented model', args.name)

    return rep_model
