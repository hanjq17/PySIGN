from .scalar import *
from .painn import *
from .et import *
from .scalar.schnet import SchNet
from .scalar.dimenet import DimeNet
from .irreps.irreps import TFN, SE3Transformer


def get_model_from_args(node_dim, edge_attr_dim, args, dynamics=False):

    if args.model == 'EGNN':
        rep_model = EGNN(in_node_nf=node_dim, hidden_nf=args.hidden_dim, out_node_nf=args.hidden_dim,
                         in_edge_nf=edge_attr_dim, n_layers=args.n_layers, use_vel=dynamics)
    elif args.model == 'RF':
        rep_model = RadialField(hidden_nf=args.hidden_dim, edge_attr_nf=edge_attr_dim, n_layers=args.n_layers)
    elif args.model == 'SchNet':
        rep_model = SchNet(in_node_nf=node_dim, out_node_nf=args.hidden_dim, hidden_nf=args.hidden_dim)
    elif args.model == 'DimeNet':
        rep_model = DimeNet(in_node_nf=node_dim, out_node_nf=args.hidden_dim, hidden_nf=args.hidden_dim,
                            num_blocks=args.n_layers)
    elif args.model == 'PaiNN':
        rep_model = PaiNN(max_z=node_dim, n_atom_basis=args.hidden_dim, n_interactions=args.n_layers)
    elif args.model == 'ET':
        rep_model = EquivariantTransformer(max_z=node_dim, hidden_channels=args.hidden_dim, num_layers=args.n_layers)
    elif args.model == 'TFN':
        rep_model = TFN(nf=args.hidden_dim // 2, n_layers=args.n_layers, num_degrees=2)
    elif args.model == 'SE3Transformer':
        rep_model = SE3Transformer(nf=args.hidden_dim // 2, n_layers=args.n_layers, num_degrees=2)
    else:
        raise NotImplementedError('Not implemented model', args.model)

    return rep_model
