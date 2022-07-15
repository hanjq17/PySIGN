import sys
sys.path.append('./')
from pysign.dataset import NBody
from pysign.nn.model import get_model_from_args
from pysign.utils import load_params, set_seed
from pysign.trainer import DynamicsTrainer
from pysign.task import Prediction
from pysign.utils.transforms import NBody_Transform
from torch_geometric.loader import DataLoader
import torch
import os
import numpy as np
import pickle

torch.cuda.set_device(0)

param_path = 'examples/configs/nbody_dynamics/egnn.yaml'
args = load_params(param_path)
set_seed(args.trainer.seed)

os.makedirs(args.data.data_path, exist_ok=True)
dataset = NBody(root=args.data.data_path, transform=NBody_Transform,
                n_particle=args.data.n_particle, num_samples=args.data.num_samples, T=args.data.T,
                sample_freq=args.data.sample_freq, num_workers=20,
                initial_step=args.task.initial_step, pred_step=args.task.pred_step)
print('Data ready')

datasets = dataset.default_split()
dataloaders = {split: DataLoader(datasets[split], batch_size=args.trainer.batch_size,
                                 shuffle=True if split == 'train' else False) for split in datasets}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rep_model = get_model_from_args(node_dim=1, edge_attr_dim=1, args=args.model, dynamics=True)

args.trainer.model_save_path = os.path.join(args.trainer.model_save_path, args.model.name)

task = Prediction(rep=rep_model, rep_dim=args.hidden_dim, output_dim=1, vector_method='diff',
                  target='vector', dynamics=True)
trainer = DynamicsTrainer(dataloaders=dataloaders, task=task, args=args.trainer,
                          device=device, lower_is_better=True, test=True, save_pred=args.trainer.save_pred)

trainer.loop()

test_dataset = datasets.get('test')
test_dataset.mode = 'rollout'
test_dataset.rollout_step = 30
trainer.model_saver.load(epoch='best')

all_loss, all_pred = trainer.evaluate_rollout_multi_system(valid=False)

temp_all_loss = [np.mean(all_loss[i]) for i in range(all_loss.shape[0])]
print('Average Rollout MSE:', np.mean(temp_all_loss), np.std(temp_all_loss))

out_dir = os.path.join(args.trainer.model_save_path, args.model.name)
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'eval_result.pkl'), 'wb') as f:
    pickle.dump((all_loss, all_pred), f)
print('Saved to', os.path.join(out_dir, 'eval_result.pkl'))


