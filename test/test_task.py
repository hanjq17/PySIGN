import sys
sys.path.append('./')
from pysign.dataset import NBody
from pysign.nn.model import get_model_from_args
from pysign.utils import load_params, set_seed
from pysign.trainer import Trainer, DynamicsTrainer
from pysign.task import Prediction, Contrastive
from pysign.utils.transforms import NBody_Transform
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch
import os
import numpy as np

param_path = 'examples/configs/test/nbody_test.yaml'
args = load_params(param_path)
set_seed(args.trainer.seed)

os.makedirs(args.data.data_path, exist_ok=True)
dataset = NBody(root=args.data.data_path, transform=NBody_Transform,
                n_particle=args.data.n_particle, num_samples=args.data.num_samples, T=args.data.T,
                sample_freq=args.data.sample_freq,
                num_workers=20, initial_step=args.task.initial_step, pred_step=args.task.pred_step)

n_train, n_val, n_test = int(args.data.num_samples // 2), int(args.data.num_samples // 4), int(args.data.num_samples // 4)
print('Data ready')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_save_path = args.trainer.model_save_path


class RadiusLabel(object):
    def __call__(self, data):
        m = data.x - data.x.mean(dim=-1, keepdim=True)
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
        return data1, data2


class EnergyForce(object):
    def __call__(self, data):
        data.y = torch.randn(1)
        data.dy = torch.randn_like(data.x)
        return data


def _prediction_test(model):
    dataset.transform = T.Compose([NBody_Transform, RadiusLabel()])
    datasets = dataset.get_split_by_num(n_train, n_val, n_test)
    dataloaders = {split: DataLoader(datasets[split], batch_size=args.trainer.batch_size,
                                     shuffle=True if split == 'train' else False)
                   for split in datasets}

    args.model.name = model

    rep_model = get_model_from_args(node_dim=1, edge_attr_dim=1, args=args.model, dynamics=True)

    args.trainer.model_save_path = os.path.join(model_save_path, 'prediction', args.model.name)

    task = Prediction(rep=rep_model, output_dim=1, rep_dim=args.model.hidden_dim, task_type='Regression', loss='MAE',
                      decoding='MLP', vector_method=None, scalar_pooling='sum', target='scalar', return_outputs=False)
    trainer = Trainer(dataloaders=dataloaders, task=task, args=args.trainer, device=device, lower_is_better=True)
    trainer.loop()


def _dynamics_test(model, decoding, vector_method):
    if vector_method == 'diff':
        decoding = None

    dataset.transform = NBody_Transform
    datasets = dataset.get_split_by_num(n_train, n_val, n_test)
    dataloaders = {split: DataLoader(datasets[split], batch_size=args.trainer.batch_size,
                                     shuffle=True if split == 'train' else False)
                   for split in datasets}

    args.model.name = model

    rep_model = get_model_from_args(node_dim=1, edge_attr_dim=1, args=args.model, dynamics=True)

    args.trainer.model_save_path = os.path.join(model_save_path,
                                                'dynamics', '_'.join([args.model.name,
                                                                      decoding if decoding is not None else '',
                                                                      vector_method if vector_method is not None else '']))

    task = Prediction(rep=rep_model, output_dim=1, rep_dim=args.model.hidden_dim, task_type='Regression', loss='MAE',
                      decoding=decoding, vector_method=vector_method, target='vector', dynamics=True,
                      return_outputs=True)

    trainer = DynamicsTrainer(dataloaders=dataloaders, task=task, args=args.trainer, device=device, lower_is_better=True,
                              test=True, save_pred=args.trainer.save_pred)

    trainer.loop()

    if args.trainer.test:
        test_dataset = datasets.get('test')
        test_dataset.mode = 'rollout'
        test_dataset.rollout_step = 10
        trainer.model_saver.load(epoch='best')

        all_loss, all_pred = trainer.evaluate_rollout_multi_system(valid=False)

        temp_all_loss = [np.mean(all_loss[i]) for i in range(all_loss.shape[0])]
        print('Average Rollout MSE:', np.mean(temp_all_loss), np.std(temp_all_loss))


def _contrastive_test(model):
    dataset.transform = T.Compose([NBody_Transform, PseudoPair()])
    datasets = dataset.get_split_by_num(n_train, n_val, n_test)
    dataloaders = {split: DataLoader(datasets[split], batch_size=args.trainer.batch_size,
                                     shuffle=True if split == 'train' else False)
                   for split in datasets}

    args.model.name = model

    rep_model = get_model_from_args(node_dim=1, edge_attr_dim=1, args=args.model, dynamics=True)

    args.trainer.model_save_path = os.path.join(model_save_path, 'contrastive', args.model.name)

    task = Contrastive(rep=rep_model, output_dim=1, rep_dim=args.model.hidden_dim, task_type='BinaryClassification',
                       loss='BCE',
                       return_outputs=True, dynamics=False)
    trainer = Trainer(dataloaders=dataloaders, task=task, args=args.trainer, device=device, lower_is_better=True)

    trainer.loop()


def _energyforce_test(model, decoding, vector_method):
    dataset.transform = T.Compose([NBody_Transform, EnergyForce()])

    datasets = dataset.get_split_by_num(n_train, n_val, n_test)
    dataloaders = {split: DataLoader(datasets[split], batch_size=args.trainer.batch_size,
                                     shuffle=True if split == 'train' else False)
                   for split in datasets}

    args.model.name = model

    rep_model = get_model_from_args(node_dim=1, edge_attr_dim=1, args=args.model, dynamics=True)

    args.trainer.model_save_path = os.path.join(model_save_path,
                                                'energyforce', '_'.join([args.model.name,
                                                                         decoding if decoding is not None else '',
                                                                         vector_method if vector_method is not None else '']))

    task = Prediction(rep=rep_model, rep_dim=args.model.hidden_dim, output_dim=1, task_type='Regression',
                      loss='MAE', decoding=decoding, vector_method=vector_method, scalar_pooling='sum',
                      target=['scalar', 'vector'], loss_weight=[0.2, 0.8], return_outputs=False, dynamics=False)
    trainer = Trainer(dataloaders=dataloaders, task=task, args=args.trainer, device=device, lower_is_better=True, test=True)

    trainer.loop()


def test_task():
    model_map = {
        'TFN': [('MLP', 'diff')],
        'SE3_Tr': [('MLP', 'diff')],
        'SchNet': [('MLP', 'gradient')],
        'EGNN': [('MLP', 'diff'), ('MLP', 'gradient')],
        'RF': [('MLP', 'diff')],
        'PaiNN': [('MLP', 'diff'), ('GatedBlock', None)],
        'ET': [('MLP', 'diff'), ('GatedBlock', None)],
    }

    for model in model_map:

        if model not in ['RF']:
            print("=" * 5, f"Prediction Test of {model}", "=" * 5)
            _prediction_test(model)
            print("=" * 5, f"Contrastive Test of {model}", "=" * 5)
            _contrastive_test(model)

        for decoder in model_map[model]:
            print("=" * 5, f"Dynamics Test of {model} & {decoder}", "=" * 5)
            _dynamics_test(model, *decoder)
            if model not in ['RF']:
                print("=" * 5, f"Energy & Force Test of {model} & {decoder}", "=" * 5)
                _energyforce_test(model, *decoder)
