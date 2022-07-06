import sys
sys.path.append('./')
from pysign.dataset import Atom3DDataset
from pysign.nn.model import get_model_from_args
from pysign.utils import get_default_args, load_params, set_seed
from pysign.trainer import PredictionTrainer
from pysign.task import Contrastive
from pysign.utils.transforms import LEP_Transform
from torch_geometric.loader import DataLoader
import torch

torch.cuda.set_device(0)

param_path = 'examples/configs/atom3d_lep_config.json'
args = get_default_args()
args = load_params(args, param_path=param_path)
set_seed(args.seed)

transform = LEP_Transform()
datasets = {
    'train': Atom3DDataset(root=args.data_path, task='lep', split='train', dataset_arg='protein', transform=transform),
    'val': Atom3DDataset(root=args.data_path, task='lep', split='val', dataset_arg='protein', transform=transform),
    'test': Atom3DDataset(root=args.data_path, task='lep', split='test', dataset_arg='protein', transform=transform)
}

print('Data ready')

dataloaders = {split: DataLoader(datasets[split], batch_size=args.batch_size, shuffle=True if split == 'train' else False)
               for split in datasets}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rep_model = get_model_from_args(node_dim=18, edge_attr_dim=1, args=args)

task = Contrastive(rep=rep_model, output_dim=1, rep_dim=args.hidden_dim, task_type='BinaryClassification', loss='BCE')
trainer = PredictionTrainer(dataloaders=dataloaders, task=task, args=args, device=device, lower_is_better=True)

trainer.loop()

