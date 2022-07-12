import sys
sys.path.append('./')
from pysign.dataset import MD17_Dynamics
from pysign.nn.model import get_model_from_args
from pysign.utils import load_params, set_seed
from pysign.trainer import DynamicsTrainer
from pysign.task import Prediction
from pysign.utils.transforms import MD17_Transform
from torch_geometric.loader import DataLoader
import torch
import os
import numpy as np
import pickle

torch.cuda.set_device(0)

param_path = 'examples/configs/md17_dynamics/egnn.yaml'
args = load_params(param_path)
set_seed(args.trainer.seed)

transform = MD17_Transform(max_atom_type=args.task.max_atom_type, charge_power=args.task.charge_power, 
                           atom_type_name='charge', cutoff=1.6, max_hop=args.task.max_hop)
base_path = os.path.join(args.data.data_path, args.data.molecule)
os.makedirs(base_path, exist_ok=True)
dataset = MD17_Dynamics(root=base_path, dataset_arg=args.data.molecule, pred_step=args.task.pred_step)
transform.get_example(dataset[0])
dataset.transform = transform
print('Data ready')

datasets = dataset.default_split()
dataloaders = {split: DataLoader(datasets[split], batch_size=args.trainer.batch_size,
                                 shuffle=True if split == 'train' else False) for split in datasets}
# dataloaders['test'].batch_size = 1  # Single trajectory does not support batch size > 1 now.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rep_model = get_model_from_args(node_dim=args.task.max_atom_type * (args.task.charge_power + 1), edge_attr_dim=0,
                                args=args.model, dynamics=True)

args.trainer.model_save_path = os.path.join(args.trainer.model_save_path, args.model.name, args.data.molecule)

task = Prediction(rep=rep_model, rep_dim=args.model.hidden_dim, output_dim=1,
                  loss='MAE', vector_method='diff', target='vector', dynamics=True, return_outputs=True)
trainer = DynamicsTrainer(dataloaders=dataloaders, task=task, args=args.trainer, device=device, lower_is_better=True,
                          test=False, rollout_step=args.task.rollout_step, save_pred=args.trainer.save_pred)

trainer.loop()

trainer.model_saver.load(epoch='best')

all_loss, all_pred = trainer.evaluate_rollout(valid=False)

temp_all_loss = [np.mean(all_loss[i]) for i in range(all_loss.shape[0])]
print('Average Rollout MSE:', np.mean(temp_all_loss), np.std(temp_all_loss))

out_dir = os.path.join(args.trainer.eval_result_path, args.model.name, args.data.molecule)
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'eval_result.pkl'), 'wb') as f:
    pickle.dump((all_loss, all_pred), f)
print('Saved to', os.path.join(out_dir, 'eval_result.pkl'))


