import sys
sys.path.append('./')
from airgeom.dataset import MD17
from airgeom.nn.model import get_model_from_args
from airgeom.utils import get_default_args, load_params, set_seed
from airgeom.trainer import MultiTaskTrainer
from airgeom.task import EnergyForcePrediction
from airgeom.utils.transforms import MD17_Transform
from torch_geometric.loader import DataLoader
import torch
import os

torch.cuda.set_device(0)

param_path = 'examples/configs/md17_config.json'
args = get_default_args()
args = load_params(args, param_path=param_path)
set_seed(args.seed)

transform = MD17_Transform(max_atom_type=args.max_atom_type, charge_power=args.charge_power, atom_type_name='z',
                           cutoff=1.6, max_hop=args.max_hop)
base_path = os.path.join(args.data_path, args.molecule)
os.makedirs(base_path, exist_ok=True)
dataset = MD17(root=base_path, dataset_arg=args.molecule)
transform.get_example(dataset[0])
dataset.transform = transform
print('Data ready')

datasets = dataset.default_split()
dataloaders = {split: DataLoader(datasets[split], batch_size=args.batch_size, shuffle=True if split == 'train' else False)
               for split in datasets}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rep_model = get_model_from_args(node_dim=args.max_atom_type * (args.charge_power + 1), edge_attr_dim=0, args=args)

args.model_save_path = os.path.join(args.model_save_path, '_'.join([args.model, args.decoder]), args.molecule)

task = EnergyForcePrediction(rep=rep_model, rep_dim=args.hidden_dim, decoder_type=args.decoder,
                             mean=dataset.mean(), std=dataset.std(), loss='MAE')
trainer = MultiTaskTrainer(dataloaders=dataloaders, task=task, args=args, device=device, lower_is_better=True, test=False)

trainer.loop()

trainer.model_saver.load(epoch='best')
test_result = trainer.evaluate()
print('Test', end=' ')
for metric in test_result:
    print('{}: {:.6f}'.format(metric, test_result[metric]), end=' ')
print()
