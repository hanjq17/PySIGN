import sys
sys.path.append('./')
from pysign.dataset import MD17
from pysign.nn.model import get_model_from_args
from pysign.utils import load_params, set_seed
from pysign.trainer import Trainer
from pysign.task import Prediction
from pysign.utils.transforms import MD17_Transform
from torch_geometric.loader import DataLoader
import torch
import os

torch.cuda.set_device(0)

param_path = 'examples/configs/md17_ef/egnn.yaml'
args = load_params(param_path)
set_seed(args.trainer.seed)

transform = MD17_Transform(max_atom_type=args.task.max_atom_type, charge_power=args.task.charge_power,
                           atom_type_name='charge', cutoff=1.6, max_hop=args.task.max_hop)
base_path = os.path.join(args.data.data_path, args.data.molecule)
os.makedirs(base_path, exist_ok=True)
dataset = MD17(root=base_path, dataset_arg=args.data.molecule)
transform.get_example(dataset[0])
dataset.transform = transform
print('Data ready')

datasets = dataset.default_split()
dataloaders = {split: DataLoader(datasets[split], batch_size=args.trainer.batch_size,
                                 shuffle=True if split == 'train' else False) for split in datasets}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rep_model = get_model_from_args(node_dim=args.task.max_atom_type * (args.task.charge_power + 1),
                                edge_attr_dim=0, args=args.model)

args.trainer.model_save_path = os.path.join(args.trainer.model_save_path, args.model.name, args.data.molecule)

task = Prediction(rep=rep_model, rep_dim=args.model.hidden_dim, normalize=(dataset.mean(), dataset.std()), loss='MAE',
                  output_dim=1, scalar_pooling='sum', decoding='MLP', vector_method='grad', target=['scalar', 'vector'],
                  loss_weight=[0.2, 0.8])
trainer = Trainer(dataloaders=dataloaders, task=task, args=args.trainer, device=device, lower_is_better=True, test=False)

trainer.loop()

trainer.model_saver.load(epoch='best')
test_result = trainer.evaluate()
print('Test', end=' ')
for metric in test_result:
    print('{}: {:.6f}'.format(metric, test_result[metric]), end=' ')
print()
