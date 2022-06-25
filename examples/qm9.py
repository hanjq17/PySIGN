import sys
sys.path.append('./')
from airgeom.dataset import QM9
from airgeom.nn.model import EGNN, PaiNN, EquivariantTransformer, RadialField, SchNet, DimeNet
from airgeom.utils import get_default_args, load_params, ToFullyConnected, set_seed, AtomOnehot
from airgeom.trainer import PredictionTrainer
from airgeom.task import Prediction
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import torch
import numpy as np
torch.cuda.set_device(0)

param_path = 'examples/configs/qm9_config.json'
args = get_default_args()
args = load_params(args, param_path=param_path)
set_seed(args.seed)

transform = T.Compose([ToFullyConnected(preserve_edge_attr=False), AtomOnehot(max_atom_type=10, charge_power=2, atom_type_name='z', atom_list=[1,6,7,8,9])])
dataset = QM9(root=args.data_path, task=args.target, transform=transform).shuffle()
print('Data ready')
# Normalize targets to mean = 0 and std = 1.
mean = dataset.mean()
std = dataset.std()

# Split datasets.

# Now generate random permutations to assign molecules to training/validation/test sets.
Nmols = len(dataset)

Ntrain = 100000
Ntest = int(0.1*Nmols)
Nvalid = Nmols - (Ntrain + Ntest)

# Generate random permutation
np.random.seed(0)
data_perm = np.random.permutation(Nmols)

# Now use the permutations to generate the indices of the dataset splits.
# train, valid, test, extra = np.split(included_idxs[data_perm], [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])
train, valid, test, extra = np.split(
    data_perm, [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

test_dataset = dataset[test]
val_dataset = dataset[valid]
train_dataset = dataset[train]
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rep_model = EGNN(in_node_nf=15, hidden_nf=args.hidden_dim, out_node_nf=args.hidden_dim, in_edge_nf=0,
                  n_layers=args.n_layers)
# rep_model = RadialField(hidden_nf=args.hidden_dim, edge_attr_nf=5, n_layers=args.n_layers)
# rep_model = PaiNN(max_z=11, n_atom_basis=args.hidden_dim, n_interactions=args.n_layers)
# rep_model = EquivariantTransformer(max_z=11, hidden_channels=args.hidden_dim, num_layers=args.n_layers)
# rep_model = SchNet(in_node_nf=11, out_node_nf=args.hidden_dim, hidden_nf=args.hidden_dim)
# Grad clip is needed if num_blocks is set to larger value
# rep_model = DimeNet(in_node_nf=11, out_node_nf=args.hidden_dim, hidden_nf=64, num_blocks=1)

task = Prediction(rep=rep_model, output_dim=1, rep_dim=args.hidden_dim, task_type='Regression', loss='MAE', mean=mean, std=std)
trainer = PredictionTrainer(dataloaders=dataloaders, task=task, args=args, device=device, lower_is_better=True)

trainer.loop()

