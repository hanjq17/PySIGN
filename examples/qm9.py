import sys
sys.path.append('./')
from pysign.dataset import QM9
from pysign.nn.model import get_model_from_args
from pysign.utils import load_params, set_seed
from pysign.trainer import Trainer
from pysign.task import Prediction
from pysign.utils.transforms import QM9_Transform
from torch_geometric.loader import DataLoader
import torch

param_path = 'examples/configs/qm9/egnn.yml'
args = load_params(param_path=param_path)
set_seed(args.trainer.seed)

dataset = QM9(root=args.data.data_path, task=args.data.target, transform=QM9_Transform(args.task.charge_power))
print('Data ready')

datasets = dataset.default_split()
dataloaders = {split: DataLoader(datasets[split], batch_size=args.trainer.batch_size,
                                 shuffle=True if split == 'train' else False)
               for split in datasets}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rep_model = get_model_from_args(node_dim=5 * (args.task.charge_power + 1), edge_attr_dim=0, args=args.model)

task = Prediction(rep=rep_model, output_dim=1, rep_dim=args.model.hidden_dim, task_type='Regression', loss='MAE',
                  normalize=(dataset.mean(), dataset.std()))
trainer = Trainer(dataloaders=dataloaders, task=task, args=args.trainer, device=device, lower_is_better=True)

trainer.loop()

