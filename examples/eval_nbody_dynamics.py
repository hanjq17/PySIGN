import sys

sys.path.append('./')
from airgeom.dataset import NBody
from airgeom.nn.model import EGNN, PaiNN, EquivariantTransformer, RadialField, SchNet, DimeNet, TFN, SE3Transformer
from airgeom.utils import get_default_args, load_params, ToFullyConnected, set_seed
from airgeom.trainer import DynamicsTrainer
from airgeom.task import DynamicsPrediction
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import torch
import os
import numpy as np
import pickle

torch.cuda.set_device(0)

param_path = 'examples/configs/nbody_config.json'
args = get_default_args()
args = load_params(args, param_path=param_path)
set_seed(args.seed)


class NBody_transform(object):
    def __call__(self, data):
        data.edge_attr = data.charge[data.edge_index[0]] * data.charge[data.edge_index[1]]
        data.x = torch.norm(data.v, dim=-1, keepdim=True)
        data['z'] = data.charge  # for TFN and SE3-Tr.
        return data


os.makedirs(args.data_path, exist_ok=True)
dataset = NBody(root=args.data_path, transform=T.Compose([ToFullyConnected(), NBody_transform()]),
                n_particle=args.n_particle, num_samples=args.num_samples, T=args.T, sample_freq=args.sample_freq,
                num_workers=20, initial_step=args.initial_step, pred_step=args.pred_step)
# Change the mode of dataset from one-step to roll-out for rollout evaluation
dataset.mode = 'rollout'
dataset.rollout_step = 30

print('Data ready')

# Split datasets
test_dataset = dataset[900: 1000]

test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

dataloaders = {'train': None, 'val': None, 'test': test_loader}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.model == 'EGNN':
    rep_model = EGNN(in_node_nf=1, hidden_nf=args.hidden_dim,
                     out_node_nf=args.hidden_dim, in_edge_nf=1,
                     n_layers=args.n_layers, use_vel=True)
elif args.model == 'RF':
    rep_model = RadialField(hidden_nf=args.hidden_dim, edge_attr_nf=1, n_layers=args.n_layers)

elif args.model == 'TFN':
    rep_model = TFN(nf=args.hidden_dim // 2, n_layers=args.n_layers, num_degrees=2)

elif args.model == 'SE3Transformer':
    rep_model = SE3Transformer(nf=args.hidden_dim // 2, n_layers=args.n_layers, num_degrees=2)

elif args.model == 'SchNet':
    rep_model = SchNet(in_node_nf=1, out_node_nf=args.hidden_dim,
                       hidden_nf=args.hidden_dim)

elif args.model == 'PaiNN':
    rep_model = PaiNN(max_z=1, n_atom_basis=args.hidden_dim,
                      n_interactions=args.n_layers)

elif args.model == 'ET':
    rep_model = EquivariantTransformer(max_z=1,
                                       hidden_channels=args.hidden_dim, num_layers=args.n_layers)

else:
    raise NotImplementedError('Unknown model', args.model)

args.model_save_path = os.path.join(args.model_save_path, args.model)

task = DynamicsPrediction(rep=rep_model, rep_dim=args.hidden_dim, decoder_type=args.decoder)
trainer = DynamicsTrainer(dataloaders=dataloaders, task=task, args=args, device=device, lower_is_better=True, test=False,
                          save_pred=args.save_pred)
trainer.model_saver.load(epoch='best')

all_loss, all_pred = trainer.evaluate_rollout_multi_system(valid=False)

temp_all_loss = [np.mean(all_loss[i]) for i in range(all_loss.shape[0])]
print('Average Rollout MSE:', np.mean(temp_all_loss), np.std(temp_all_loss))

out_dir = os.path.join(args.eval_result_path, args.model)
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'eval_result.pkl'), 'wb') as f:
    pickle.dump((all_loss, all_pred), f)
print('Saved to', os.path.join(out_dir, 'eval_result.pkl'))

