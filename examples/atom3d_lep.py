import sys
sys.path.append('./')
from airgeom.dataset import Atom3DDataset
from airgeom.nn.model import EGNN, PaiNN, EquivariantTransformer, RadialField, SchNet, DimeNet
from airgeom.utils import get_default_args, load_params, set_seed
from airgeom.trainer import PredictionTrainer
from airgeom.task import Contrastive
from torch_geometric.loader import DataLoader
import torch

torch.cuda.set_device(0)

param_path = 'examples/configs/atom3d_lep_config.json'
args = get_default_args()
args = load_params(args, param_path=param_path)
set_seed(args.seed)


class LEPProcess(object):
    def __call__(self, data):
        data1, data2 = data
        data1.edge_attr = data1.edge_attr.reshape(-1,1)
        data2.edge_attr = data2.edge_attr.reshape(-1,1)
        if isinstance(data1.y, str):
            data1.y = torch.FloatTensor([data1.y == 'A']) 
        return data1, data2


transform = LEPProcess()
train_dataset = Atom3DDataset(root=args.data_path, task='lep', split='train', dataset_arg='protein', transform=transform)
val_dataset = Atom3DDataset(root=args.data_path, task='lep', split='val', dataset_arg='protein', transform=transform)
test_dataset = Atom3DDataset(root=args.data_path, task='lep', split='test', dataset_arg='protein', transform=transform)

print('Data ready')

# Split datasets.
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rep_model = EGNN(in_node_nf=18, hidden_nf=args.hidden_dim, out_node_nf=args.hidden_dim, in_edge_nf=1,
                 n_layers=args.n_layers)
# rep_model = RadialField(hidden_nf=args.hidden_dim, edge_attr_nf=5, n_layers=args.n_layers)
# rep_model = PaiNN(max_z=11, n_atom_basis=args.hidden_dim, n_interactions=args.n_layers)
# rep_model = EquivariantTransformer(max_z=11, hidden_channels=args.hidden_dim, num_layers=args.n_layers)
# rep_model = SchNet(in_node_nf=11, out_node_nf=args.hidden_dim, hidden_nf=args.hidden_dim)
# Grad clip is needed if num_blocks is set to larger value
# rep_model = DimeNet(in_node_nf=11, out_node_nf=args.hidden_dim, hidden_nf=64, num_blocks=1)

task = Contrastive(rep=rep_model, output_dim=1, rep_dim=args.hidden_dim, task_type='BinaryClassification', loss='BCE')
trainer = PredictionTrainer(dataloaders=dataloaders, task=task, args=args, device=device, lower_is_better=True)

trainer.loop()

