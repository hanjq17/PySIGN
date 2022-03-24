import sys
sys.path.append('./')
from airgeom.dataset import QM9
from airgeom.nn.model import EGNN, PaiNN, EquivariantTransformer, RadialField
from airgeom.utils import get_default_args, load_params, ToFullyConnected, set_seed
from airgeom.trainer import Trainer
from airgeom.task import Prediction
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import torch

torch.cuda.set_device(0)

param_path = 'examples/configs/qm9_config.json'
args = get_default_args()
args = load_params(args, param_path=param_path)
set_seed(args.seed)


class SelectTarget(object):
    def __call__(self, data):
        data.y = data.y[:, args.target]
        return data


transform = T.Compose([SelectTarget(), ToFullyConnected(), T.Distance(norm=False)])
dataset = QM9(root=args.data_path, transform=transform).shuffle()
print('Data ready')
# Normalize targets to mean = 0 and std = 1.
mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean[:, args.target].item(), std[:, args.target].item()

# Split datasets.
test_dataset = dataset[:10000]
val_dataset = dataset[10000:20000]
train_dataset = dataset[20000:]
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rep_model = EGNN(in_node_nf=11, hidden_nf=args.hidden_dim, out_node_nf=args.hidden_dim, in_edge_nf=5,
                  n_layers=args.n_layers)
# rep_model = RadialField(hidden_nf=args.hidden_dim, edge_attr_nf=5, n_layers=args.n_layers)
# rep_model = PaiNN(max_z=11, n_atom_basis=args.hidden_dim, n_interactions=args.n_layers)
# rep_model = EquivariantTransformer(max_z=11, hidden_channels=args.hidden_dim, num_layers=args.n_layers)

task = Prediction(rep=rep_model, output_dim=1, rep_dim=args.hidden_dim)
trainer = Trainer(dataloaders=dataloaders, task=task, args=args, device=device, lower_is_better=True)

trainer.loop()

