import sys
sys.path.append('./')
from pysign.dataset import QM9
from pysign.nn.model import get_model_from_args
from pysign.utils import get_default_args, load_params, set_seed
from pysign.trainer import PredictionTrainer
from pysign.task import Prediction
from pysign.utils.transforms import QM9_Transform
from torch_geometric.loader import DataLoader
import torch

param_path = 'examples/configs/qm9_config.json'
args = get_default_args()
args = load_params(args, param_path=param_path)
set_seed(args.seed)

dataset = QM9(root=args.data_path, task=args.target, transform=QM9_Transform)
print('Data ready')

datasets = dataset.default_split()
dataloaders = {split: DataLoader(datasets[split], batch_size=args.batch_size, shuffle=True if split == 'train' else False)
               for split in datasets}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rep_model = get_model_from_args(node_dim=15, edge_attr_dim=0, args=args)

task = Prediction(rep=rep_model, output_dim=1, rep_dim=args.hidden_dim, task_type='Regression', loss='MAE',
                  mean=dataset.mean(), std=dataset.std())
trainer = PredictionTrainer(dataloaders=dataloaders, task=task, args=args, device=device, lower_is_better=True)

trainer.loop()

