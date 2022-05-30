import sys
sys.path.append('./')
from airgeom.dataset import MD17, MD17_trajectory
from airgeom.nn.model import EGNN, PaiNN, EquivariantTransformer, RadialField, SchNet, DimeNet
from airgeom.utils import get_default_args, load_params, ToFullyConnected, set_seed, AtomOnehot
from airgeom.trainer import Trainer
from airgeom.task import TrajectoryPrediction
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import torch
import os

torch.cuda.set_device(0)

param_path = 'examples/configs/md17_config.json'
args = get_default_args()
args = load_params(args, param_path=param_path)
set_seed(args.seed)

args.batch_size = 32


transform = T.Compose([AtomOnehot(max_atom_type=args.max_atom_type, charge_power=args.charge_power, atom_type_name='z'),ToFullyConnected()])
base_path = os.path.join(args.data_path, args.molecule)
os.makedirs(base_path, exist_ok=True)
dataset = MD17_trajectory(root=base_path, dataset_arg=args.molecule, transform=transform, vel_step=args.vel_step, pred_step=args.pred_step)
print('Data ready')

# Split datasets.
train_dataset = dataset[args.vel_step: args.vel_step + 9500]
val_dataset = dataset[args.vel_step + 9500: args.vel_step + 10000]
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
dataloaders = {'train': train_loader, 'val': val_loader, 'test': None}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rep_model = EGNN(in_node_nf=args.max_atom_type * (args.charge_power + 1), hidden_nf=args.hidden_dim, out_node_nf=args.hidden_dim, in_edge_nf=0,
                 n_layers=args.n_layers, use_vel=True)
# rep_model = RadialField(hidden_nf=args.hidden_dim, edge_attr_nf=0, n_layers=args.n_layers)
# rep_model = PaiNN(max_z=11, n_atom_basis=args.hidden_dim, n_interactions=args.n_layers)
# rep_model = EquivariantTransformer(max_z=11, hidden_channels=args.hidden_dim, num_layers=args.n_layers)
# rep_model = SchNet(in_node_nf=11, out_node_nf=args.hidden_dim, hidden_nf=args.hidden_dim)
# Grad clip is needed if num_blocks is set to larger value
# rep_model = DimeNet(in_node_nf=11, out_node_nf=args.hidden_dim, hidden_nf=64, num_blocks=1)

task = TrajectoryPrediction(rep=rep_model, rep_dim=args.hidden_dim, decoder_type='DifferentialVector')
trainer = Trainer(dataloaders=dataloaders, task=task, args=args, device=device, lower_is_better=True, test=False)

trainer.loop()

