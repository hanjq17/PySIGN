import sys
sys.path.append('./')
from airgeom.dataset import NBody
from airgeom.nn.model import get_model_from_args
from airgeom.utils import get_default_args, load_params, set_seed
from airgeom.trainer import DynamicsTrainer, PredictionTrainer, MultiTaskTrainer
from airgeom.task import DynamicsPrediction, Prediction, Contrastive, EnergyForcePrediction
from airgeom.utils.transforms import NBody_Transform
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch
import os
import numpy as np
import pickle

# torch.cuda.set_device(0)

param_path = 'examples/configs/nbody_test_config.json'
args = get_default_args()
args = load_params(args, param_path=param_path)
set_seed(args.seed)

os.makedirs(args.data_path, exist_ok=True)
dataset = NBody(root=args.data_path, transform=NBody_Transform,
                n_particle=args.n_particle, num_samples=args.num_samples, T=args.T, sample_freq=args.sample_freq,
                num_workers=20, initial_step=args.initial_step, pred_step=args.pred_step)

n_train, n_val, n_test = int(args.num_samples // 2), int(args.num_samples // 4), int(args.num_samples // 4)
print('Data ready')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_save_path = args.model_save_path

class RadiusLabel(object):
    def __call__(self, data):
        m = data.x - data.x.mean(dim=-1,keepdim=True)
        data.y = m.norm(dim=-1).max()
        return data

class PseudoPair(object):
    counter = 0
    def __call__(self, data):
        data1, data2 = data, data
        if self.counter % 2 == 0:
            data1.y = torch.tensor(0).float()
        else:
            data1.y = torch.tensor(1).float()
        self.counter += 1
        # data1.y = torch.randint(2,(1,)).float()
        return data1, data2    

class EnergyForce(object):
    def __call__(self, data):
        data.y = torch.randn(1)
        data.dy = torch.randn_like(data.x)
        return data

def prediction_test(model):

    dataset.transform = T.Compose([NBody_Transform, RadiusLabel()])
    datasets = dataset.get_split_by_num(n_train, n_val, n_test)
    dataloaders = {split: DataLoader(datasets[split], batch_size=args.batch_size, shuffle=True if split == 'train' else False)
                for split in datasets}

    args.model = model

    rep_model = get_model_from_args(node_dim=1, edge_attr_dim=1, args=args, dynamics=True)

    args.model_save_path = os.path.join(model_save_path, 'prediction', args.model)

    task = Prediction(rep=rep_model, output_dim=1, rep_dim=args.hidden_dim, task_type='Regression', loss='MAE')
    trainer = PredictionTrainer(dataloaders=dataloaders, task=task, args=args, device=device, lower_is_better=True)

    trainer.loop()


def dynamics_test(model, decoder):

    dataset.transform = NBody_Transform

    datasets = dataset.get_split_by_num(n_train, n_val, n_test)
    dataloaders = {split: DataLoader(datasets[split], batch_size=args.batch_size, shuffle=True if split == 'train' else False)
                for split in datasets}

    args.model = model
    args.decoder = decoder

    rep_model = get_model_from_args(node_dim=1, edge_attr_dim=1, args=args, dynamics=True)

    args.model_save_path = os.path.join(model_save_path, 'dynamics', args.model + '_' + args.decoder)

    task = DynamicsPrediction(rep=rep_model, rep_dim=args.hidden_dim, decoder_type=args.decoder)
    trainer = DynamicsTrainer(dataloaders=dataloaders, task=task, args=args, device=device, lower_is_better=True, test=True,
                            save_pred=args.save_pred)

    trainer.loop()

    if args.test:
        test_dataset = datasets.get('test')
        test_dataset.mode = 'rollout'
        test_dataset.rollout_step = 10
        trainer.model_saver.load(epoch='best')

        all_loss, all_pred = trainer.evaluate_rollout_multi_system(valid=False)

        temp_all_loss = [np.mean(all_loss[i]) for i in range(all_loss.shape[0])]
        print('Average Rollout MSE:', np.mean(temp_all_loss), np.std(temp_all_loss))

        out_dir = os.path.join(args.eval_result_path, args.model)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'eval_result.pkl'), 'wb') as f:
            pickle.dump((all_loss, all_pred), f)
        print('Saved to', os.path.join(out_dir, 'eval_result.pkl'))

def contrastive_test(model):

    dataset.transform = T.Compose([NBody_Transform, PseudoPair()])
    datasets = dataset.get_split_by_num(n_train, n_val, n_test)
    dataloaders = {split: DataLoader(datasets[split], batch_size=args.batch_size, shuffle=True if split == 'train' else False)
                for split in datasets}

    args.model = model

    rep_model = get_model_from_args(node_dim=1, edge_attr_dim=1, args=args, dynamics=True)

    args.model_save_path = os.path.join(model_save_path, 'contrastive', args.model)

    task = Contrastive(rep=rep_model, output_dim=1, rep_dim=args.hidden_dim, task_type='BinaryClassification', loss='BCE')
    trainer = PredictionTrainer(dataloaders=dataloaders, task=task, args=args, device=device, lower_is_better=True)

    trainer.loop()

def energyforce_test(model, decoder):

    dataset.transform = T.Compose([NBody_Transform, EnergyForce()])

    datasets = dataset.get_split_by_num(n_train, n_val, n_test)
    dataloaders = {split: DataLoader(datasets[split], batch_size=args.batch_size, shuffle=True if split == 'train' else False)
                for split in datasets}

    args.model = model
    args.decoder = decoder

    rep_model = get_model_from_args(node_dim=1, edge_attr_dim=1, args=args, dynamics=True)

    args.model_save_path = os.path.join(model_save_path, 'energyforce', args.model + '_' + args.decoder)

    task = EnergyForcePrediction(rep=rep_model, rep_dim=args.hidden_dim, decoder_type=args.decoder, loss='MAE')
    trainer = MultiTaskTrainer(dataloaders=dataloaders, task=task, args=args, device=device, lower_is_better=True, test=True)

    trainer.loop()

if __name__ == '__main__':

    model_map = {
        'TFN': ['DifferentialVector'],
        'SE3Transformer': ['DifferentialVector'],
        'SchNet': ['Scalar'],
        'DimeNet': ['Scalar'],
        'EGNN': ['Scalar', 'DifferentialVector'],
        'RF': ['DifferentialVector'],
        'PaiNN': ['Scalar', 'EquivariantVector'],
        'ET': ['Scalar', 'EquivariantVector']
    }

    for model in model_map:

        if model not in ['RF']:

            print("="*5, f"Prediction Test of {model}", "="*5)

            prediction_test(model)

            print("="*5, f"Contrastive Test of {model}", "="*5)

            contrastive_test(model)

        for decoder in model_map[model]:

            print("="*5, f"Dynamics Test of {model} & {decoder}", "="*5)

            dynamics_test(model, decoder)

            if model not in ['RF']:

                print("="*5, f"Energy & Force Test of {model} & {decoder}", "="*5)

                energyforce_test(model, decoder)

