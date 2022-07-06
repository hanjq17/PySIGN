import sys
sys.path.append('./')
from airgeom.dataset import QM9
from airgeom.nn.model import get_model_from_args
from airgeom.utils import get_default_args, load_params, set_seed
from airgeom.trainer import ConformationTrainer
from airgeom.task import ConformationGeneration
from airgeom.utils.transforms import QM9_Transform
from torch_geometric.loader import DataLoader
import torch
import os
import numpy as np
import pickle

torch.cuda.set_device(0)

param_path = 'examples/configs/qm9_conformation_config.json'
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

task = ConformationGeneration(rep=rep_model, rep_dim=args.hidden_dim, num_steps=args.num_steps, loss='MSE')
trainer = ConformationTrainer(dataloaders=dataloaders, task=task, args=args, device=device, lower_is_better=True)

trainer.loop()

# if args.test:
#     trainer.model_saver.load(epoch='best')
# 
#     all_loss, all_pred = trainer.evaluate_rollout(valid=False)
# 
#     temp_all_loss = [np.mean(all_loss[i]) for i in range(all_loss.shape[0])]
#     print('Average Rollout MSE:', np.mean(temp_all_loss), np.std(temp_all_loss))
# 
#     out_dir = os.path.join(args.eval_result_path, '_'.join([args.model, args.decoder]), args.molecule)
#     os.makedirs(out_dir, exist_ok=True)
#     with open(os.path.join(out_dir, 'eval_result.pkl'), 'wb') as f:
#         pickle.dump((all_loss, all_pred), f)
#     print('Saved to', os.path.join(out_dir, 'eval_result.pkl'))


