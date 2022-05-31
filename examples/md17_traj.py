import sys
sys.path.append('./')
from airgeom.dataset import MD17, MD17_trajectory
from airgeom.nn.model import EGNN, PaiNN, EquivariantTransformer, RadialField, SchNet, DimeNet, TFN, SE3Transformer
from airgeom.utils import get_default_args, load_params, ToFullyConnected, set_seed, AtomOnehot
from airgeom.trainer import Trainer
from airgeom.task import TrajectoryPrediction
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
import os

torch.cuda.set_device(0)

param_path = 'examples/configs/md17_config.json'
args = get_default_args()
args = load_params(args, param_path=param_path)
set_seed(args.seed)

class MD17_transform(object):
    def __init__(self, max_atom_type, charge_power, atom_type_name, max_hop, cutoff):
        self.max_atom_type = max_atom_type
        self.charge_power = charge_power
        self.atom_type_name = atom_type_name
        self.max_hop = max_hop
        self.cutoff = cutoff
        self.processed = False
    
    def binarize(self,x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(self, adj, order):

        adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                    self.binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]
        for i in range(2, order+1):
            adj_mats.append(self.binarize(adj_mats[i-1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)
        for i in range(1, order+1):
            order_mat += (adj_mats[i] - adj_mats[i-1]) * i
        return order_mat

    def gen_fully_connected_with_hop(self,pos):
        nodes = pos.shape[0]
        adj = torch.norm(pos.unsqueeze(0) - pos.unsqueeze(1), p=2, dim=-1) # n * n
        adj = (adj <= self.cutoff) & (~torch.eye(nodes).bool())
        adj_order = self.get_higher_order_adj_matrix(adj.long(), self.max_hop)
        # adj_order = type_mat
        fc = 1 - torch.eye(pos.shape[0], dtype=torch.long)
        ans = adj_order + fc
        edge_index, edge_type = dense_to_sparse(ans)
        return edge_index, edge_type - 1

    def gen_atom_onehot(self,atom_type):
        if self.charge_power == -1:
            return atom_type
        else:
            one_hot = F.one_hot(atom_type, self.max_atom_type)
            charge_tensor = (atom_type.unsqueeze(-1) / self.max_atom_type).pow(
                torch.arange(self.charge_power + 1., dtype=torch.float32))
            charge_tensor = charge_tensor.view(atom_type.shape + (1, self.charge_power + 1))
            atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(atom_type.shape + (-1,))
            return atom_scalars

    def get_example(self,data):
        assert hasattr(data, self.atom_type_name)
        atom_type = getattr(data, self.atom_type_name)
        self.x = self.gen_atom_onehot(atom_type)
        self.edge_index, self.edge_type = self.gen_fully_connected_with_hop(data.pos)
        self.processed = True

    def __call__(self,data):
        assert self.processed
        data.x = self.x
        data.edge_index, data.edge_type = self.edge_index, self.edge_type
        return data


transform = MD17_transform(max_atom_type=args.max_atom_type, charge_power=args.charge_power, atom_type_name='z',cutoff=1.6,max_hop=args.max_hop)
base_path = os.path.join(args.data_path, args.molecule)
os.makedirs(base_path, exist_ok=True)
dataset = MD17_trajectory(root=base_path, dataset_arg=args.molecule, vel_step=args.vel_step, pred_step=args.pred_step)
transform.get_example(dataset[0])
dataset.transform = transform
print('Data ready')

# Split datasets.
train_dataset = dataset[args.vel_step: args.vel_step + 9500]
val_dataset = dataset[args.vel_step + 9500: args.vel_step + 10000]
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
dataloaders = {'train': train_loader, 'val': val_loader, 'test': None}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.model == 'EGNN':
    rep_model = EGNN(in_node_nf=args.max_atom_type * (args.charge_power + 1), hidden_nf=args.hidden_dim, out_node_nf=args.hidden_dim, in_edge_nf=0,
                    n_layers=args.n_layers, use_vel=True)
elif args.model == 'RF':
    rep_model = RadialField(hidden_nf=args.hidden_dim, edge_attr_nf=0, n_layers=args.n_layers)
    
elif args.model == 'TFN':
    rep_model = TFN(nf=args.hidden_dim // 2, n_layers=args.n_layers, num_degrees=2)  

elif args.model == 'SE3Transformer':
    rep_model = SE3Transformer(nf=args.hidden_dim // 2, n_layers=args.n_layers, num_degrees=2)  

elif args.model == 'SchNet':
    rep_model = SchNet(in_node_nf=args.max_atom_type * (args.charge_power + 1), out_node_nf=args.hidden_dim, hidden_nf=args.hidden_dim)

elif args.model == 'PaiNN':
    rep_model = PaiNN(max_z=args.max_atom_type * (args.charge_power + 1), n_atom_basis=args.hidden_dim, n_interactions=args.n_layers)

elif args.model == 'ET':
    rep_model = EquivariantTransformer(max_z=args.max_atom_type * (args.charge_power + 1), hidden_channels=args.hidden_dim, num_layers=args.n_layers)

else:
    raise NotImplementedError('Unknown model', args.model)

# Grad clip is needed if num_blocks is set to larger value
# rep_model = DimeNet(in_node_nf=11, out_node_nf=args.hidden_dim, hidden_nf=64, num_blocks=1)

args.model_save_path = os.path.join(args.model_save_path, args.model, args.molecule)

task = TrajectoryPrediction(rep=rep_model, rep_dim=args.hidden_dim, decoder_type=args.decoder)
trainer = Trainer(dataloaders=dataloaders, task=task, args=args, device=device, lower_is_better=True, test=False)

trainer.loop()

