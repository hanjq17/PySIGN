import sys
sys.path.append('./')
from pysign.dataset import QM9
from pysign.nn.model import get_model_from_args
from pysign.utils import load_params, set_seed
from pysign.trainer import ConformationTrainer
from pysign.task import ConformationGeneration
from pysign.utils.transforms import QM9_Transform
from torch_geometric.loader import DataLoader
import torch
import os
import numpy as np
import pickle

torch.cuda.set_device(0)

param_path = 'examples/configs/qm9_conformation/egnn.yaml'
args = load_params(param_path)
set_seed(args.trainer.seed)

dataset = QM9(root=args.data.data_path, task='alpha', transform=QM9_Transform)
print('Data ready')

datasets = dataset.default_split()
dataloaders = {split: DataLoader(datasets[split], batch_size=args.trainer.batch_size,
                                 shuffle=True if split == 'train' else False) for split in datasets}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rep_model = get_model_from_args(node_dim=15 + args.model.hidden_dim, edge_attr_dim=0, args=args.model)

task = ConformationGeneration(rep=rep_model, rep_dim=args.model.hidden_dim, num_steps=args.task.num_steps, loss='MSE')
trainer = ConformationTrainer(dataloaders=dataloaders, task=task, args=args.trainer, device=device, lower_is_better=True)

print('start training')
# trainer.loop()

trainer.model_saver.load(epoch='best')

all_loss, all_pred = trainer.evaluate_rollout(valid=False)

temp_all_loss = [np.mean(all_loss[i]) for i in range(all_loss.shape[0])]
print('Average Rollout MSE:', np.mean(temp_all_loss), np.std(temp_all_loss))

out_dir = os.path.join(args.trainer.eval_result_path, args.model.name)
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'eval_result.pkl'), 'wb') as f:
    pickle.dump((all_loss, all_pred), f)
print('Saved to', os.path.join(out_dir, 'eval_result.pkl'))