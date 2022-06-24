import sys
sys.path.append('./')
from airgeom.dataset import Atom3DDataset
from airgeom.nn.model import EGNN, PaiNN, EquivariantTransformer, RadialField, SchNet, DimeNet
from airgeom.utils import get_default_args, load_params, ToFullyConnected, set_seed
from airgeom.trainer import PredictionTrainer
from airgeom.task import Prediction
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import torch

torch.cuda.set_device(0)

param_path = 'examples/configs/atom3d_lba.json'
args = get_default_args()
args = load_params(args, param_path=param_path)
set_seed(args.seed)



class SelectEdges(object):
    def __init__(self, edges_between=True):
        self.edges_between = edges_between
    def __call__(self, data):
        data.edge_attr = data.edge_attr.reshape(-1,1)
        if not self.edges_between:
            row, col = data.edge_index
            ins = data.instance
            mask = ins[row] == ins[col]
            data.edge_index = data.edge_index[:,mask]
            data.edge_attr = data.edge_attr[mask]
        return data


transform = SelectEdges()
train_dataset = Atom3DDataset(root=args.data_path, task='lba',split='train',dataset_arg='sequence-identity-30',transform=transform)
val_dataset = Atom3DDataset(root=args.data_path, task='lba',split='val',dataset_arg='sequence-identity-30',transform=transform)
test_dataset = Atom3DDataset(root=args.data_path, task='lba',split='test',dataset_arg='sequence-identity-30',transform=transform)
print('Data ready')
# Normalize targets to mean = 0 and std = 1.

# Split datasets.
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rep_model = EGNN(in_node_nf=61, hidden_nf=args.hidden_dim, out_node_nf=args.hidden_dim, in_edge_nf=1,
                  n_layers=args.n_layers)
# rep_model = RadialField(hidden_nf=args.hidden_dim, edge_attr_nf=5, n_layers=args.n_layers)
# rep_model = PaiNN(max_z=11, n_atom_basis=args.hidden_dim, n_interactions=args.n_layers)
# rep_model = EquivariantTransformer(max_z=11, hidden_channels=args.hidden_dim, num_layers=args.n_layers)
# rep_model = SchNet(in_node_nf=11, out_node_nf=args.hidden_dim, hidden_nf=args.hidden_dim)
# Grad clip is needed if num_blocks is set to larger value
# rep_model = DimeNet(in_node_nf=11, out_node_nf=args.hidden_dim, hidden_nf=64, num_blocks=1)

task = Prediction(rep=rep_model, output_dim=1, rep_dim=args.hidden_dim, task_type='Regression', loss='MSE')
trainer = PredictionTrainer(dataloaders=dataloaders, task=task, args=args, device=device, lower_is_better=True)

trainer.loop()

