from ..dataset.registry import DatasetRegistry
from ..utils import get_default_args, load_params, set_seed
from ..utils.transforms import QM9_Transform, MD17_Transform, NBody_Transform, LEP_Transform, SelectEdges
from ..nn.model import get_model_from_args
from ..trainer import PredictionTrainer, MultiTaskTrainer, DynamicsTrainer
from ..task import Prediction, EnergyForcePrediction, DynamicsPrediction
from torch_geometric.loader import DataLoader
import torch
import os
import numpy as np
import pickle


class Benchmark(object):
    def __init__(self, args_file=None):
        self.args_file = args_file
        args = get_default_args()
        self.args = load_params(args, param_path=self.args_file)
        set_seed(self.args.seed)
        self.dataset = None
        self.datasets = None
        self.encoder = None
        self._trainer = None
        self.specific_args = {}
        self.trainer_specific_args = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not hasattr(self.args, 'eval_result_path'):
            self.args.eval_result_path = self.args.model_save_path

    @property
    def dynamics(self):
        raise NotImplementedError()

    @property
    def trainer(self):
        raise NotImplementedError()

    def task(self, encoder):
        raise NotImplementedError()

    def callback_before_evaluation(self):
        return

    def launch(self):
        self.datasets = self.dataset.default_split()
        dataloaders = {
            split: DataLoader(self.datasets[split],
                              batch_size=self.args.batch_size, shuffle=True if split == 'train' else False)
            for split in self.datasets}
        encoder = get_model_from_args(args=self.args, dynamics=self.dynamics, **self.specific_args)
        trainer = self.trainer(dataloaders=dataloaders, task=self.task(encoder), args=self.args,
                               device=self.device, lower_is_better=True, **self.trainer_specific_args)
        trainer.loop()

        trainer.model_saver.load(epoch='best')

        self.callback_before_evaluation()

        if not self.dynamics:
            test_result = trainer.evaluate()
            print('Test', end=' ')
            for metric in test_result:
                print('{}: {:.6f}'.format(metric, test_result[metric]), end=' ')
            print()
        else:
            all_loss, all_pred = trainer.evaluate_rollout_multi_system(valid=False)
            temp_all_loss = [np.mean(all_loss[i]) for i in range(all_loss.shape[0])]
            print('Average Rollout MSE:', np.mean(temp_all_loss), np.std(temp_all_loss))
            os.makedirs(self.args.eval_result_path, exist_ok=True)
            with open(os.path.join(self.args.eval_result_path, 'eval_result.pkl'), 'wb') as f:
                pickle.dump((all_loss, all_pred), f)
            print('Saved to', os.path.join(self.args.eval_result_path, 'eval_result.pkl'))


class BenchmarkQM9(Benchmark):
    def __init__(self, args_file=None):
        if args_file is None:
            args_file = 'examples/configs/qm9_config.json'
        super(BenchmarkQM9, self).__init__(args_file)
        self.dataset = DatasetRegistry.get_dataset('qm9')(root=self.args.data_path,
                                                          task=self.args.target,
                                                          transform=QM9_Transform)
        print('Data ready')
        self.specific_args = {'node_dim': 15, 'edge_attr_dim': 0}
        self.trainer_specific_args = {'test': True}

    def task(self, encoder):
        task = Prediction(rep=encoder, output_dim=1, rep_dim=self.args.hidden_dim,
                          task_type='Regression', loss='MAE', mean=self.dataset.mean(), std=self.dataset.std())
        return task

    @property
    def trainer(self):
        return PredictionTrainer

    @property
    def dynamics(self):
        return False


class BenchmarkMD17(Benchmark):
    def __init__(self, args_file=None):
        if args_file is None:
            if not self.dynamics:
                args_file = 'examples/configs/md17_config.json'
            else:
                args_file = 'examples/configs/md17_dynamics_config.json'
        super(BenchmarkMD17, self).__init__(args_file)
        self.args.data_path = os.path.join(self.args.data_path, self.args.molecule)
        if not self.dynamics:
            extra_args = {}
        else:
            extra_args = {'vel_step': self.args.vel_step, 'pred_step': self.args.pred_step}
        self.dataset = DatasetRegistry.get_dataset('md17' if not self.dynamics else 'md17_dynamics')(
            root=self.args.data_path, dataset_arg=self.args.molecule, **extra_args)
        transform = MD17_Transform(max_atom_type=self.args.max_atom_type, charge_power=self.args.charge_power,
                                   atom_type_name='charge', cutoff=1.6, max_hop=self.args.max_hop)
        transform.get_example(self.dataset[0])
        self.dataset.transform = transform
        print('Data ready')
        self.specific_args = {'node_dim': self.args.max_atom_type * (self.args.charge_power + 1), 'edge_attr_dim': 0}
        self.trainer_specific_args = {'test': False}
        self.args.model_save_path = os.path.join(self.args.model_save_path,
                                                 '_'.join([self.args.model, self.args.decoder]),
                                                 self.args.molecule)
        self.args.eval_result_path = os.path.join(self.args.eval_result_path,
                                                  '_'.join([self.args.model, self.args.decoder]),
                                                  self.args.molecule)

    def task(self, encoder):
        task = EnergyForcePrediction(rep=encoder, rep_dim=self.args.hidden_dim, decoder_type=self.args.decoder,
                                     mean=self.dataset.mean(), std=self.dataset.std(), loss='MAE')
        return task

    @property
    def trainer(self):
        return MultiTaskTrainer

    @property
    def dynamics(self):
        return False


class BenchmarkMD17Dynamics(BenchmarkMD17):
    def __init__(self, args_file=None):
        super(BenchmarkMD17Dynamics, self).__init__(args_file)
        self.trainer_specific_args['rollout_step'] = self.args.rollout_step
        self.trainer_specific_args['save_pred'] = self.args.save_pred

    def task(self, encoder):
        task = DynamicsPrediction(rep=encoder, rep_dim=self.args.hidden_dim, decoder_type=self.args.decoder)
        return task

    @property
    def trainer(self):
        return DynamicsTrainer

    @property
    def dynamics(self):
        return True


class BenchmarkNBody(Benchmark):
    def __init__(self, args_file=None):
        if args_file is None:
            args_file = 'examples/configs/nbody_dynamics_config.json'
        super(BenchmarkNBody, self).__init__(args_file)
        self.dataset = DatasetRegistry.get_dataset('nbody_dynamics')(
            root=self.args.data_path, transform=NBody_Transform, n_particle=self.args.n_particle,
            num_samples=self.args.num_samples, T=self.args.T, sample_freq=self.args.sample_freq,
            num_workers=20, initial_step=self.args.initial_step, pred_step=self.args.pred_step)
        self.specific_args = {'node_dim': 1, 'edge_attr_dim': 1}
        self.trainer_specific_args = {'test': True, 'save_pred': self.args.save_pred}

    def task(self, encoder):
        task = DynamicsPrediction(rep=encoder, rep_dim=self.args.hidden_dim, decoder_type=self.args.decoder)
        return task

    @property
    def trainer(self):
        return DynamicsTrainer

    @property
    def dynamics(self):
        return True

    def callback_before_evaluation(self):
        test_dataset = self.datasets.get('test')
        test_dataset.mode = 'rollout'
        test_dataset.rollout_step = 30


class BenchmarkAtom3dLBA(Benchmark):
    def __init__(self, args_file=None):
        if args_file is None:
            args_file = 'examples/configs/atom3d_lba_config.json'
        super(BenchmarkAtom3dLBA, self).__init__(args_file)
        # TODO: add more benchmarks
