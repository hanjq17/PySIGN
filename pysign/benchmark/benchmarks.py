from ..trainer import DynamicsTrainer, Trainer
from ..task import Prediction, Contrastive
from ..utils.transforms import QM9_Transform, MD17_Transform, NBody_Transform, LEP_Transform, SelectEdges
from ..dataset.registry import DatasetRegistry
from .registry import BenchmarkRegistry
from .basic import Benchmark
import os


@BenchmarkRegistry.register_benchmark('benchmark_qm9')
class BenchmarkQM9(Benchmark):
    def __init__(self, args_file=None):
        if args_file is None:
            args_file = 'examples/configs/qm9/egnn.yaml'
        super(BenchmarkQM9, self).__init__(args_file)
        self.dataset = DatasetRegistry.get_dataset('qm9')(root=self.args.data.data_path,
                                                          task=self.args.data.target,
                                                          transform=QM9_Transform(self.args.task.charge_power),
                                                          dataset_seed=self.args.data.seed,
                                                          split=self.args.data.split)
        print('Data ready')
        self.model_specific_args = {'node_dim': 5 * (self.args.task.charge_power + 1), 'edge_attr_dim': 0}
        self.trainer_specific_args = {'test': True}
        if self.args.trainer.exp_name is None:
            self.args.trainer.model_save_path = os.path.join(self.args.trainer.model_save_path,
                                                             self.args.model.name,
                                                             self.args.data.target)

    def task(self, encoder):
        task = Prediction(rep=encoder, output_dim=1, rep_dim=self.args.model.hidden_dim,
                          task_type='Regression', loss='MAE', decoding='MLP', vector_method=None,
                          normalize=(self.dataset.mean(), self.dataset.std()), scalar_pooling='sum',
                          target='scalar', return_outputs=False, dynamics=False)
        return task

    @property
    def trainer(self):
        return Trainer

    @property
    def dynamics(self):
        return False


@BenchmarkRegistry.register_benchmark('benchmark_md17_ef')
class BenchmarkMD17(Benchmark):
    def __init__(self, args_file=None):
        if args_file is None:
            if not self.dynamics:
                args_file = 'examples/configs/md17_ef/egnn.yaml'
            else:
                args_file = 'examples/configs/md17_dynamics/egnn.yaml'
        super(BenchmarkMD17, self).__init__(args_file)
        self.args.data.data_path = os.path.join(self.args.data.data_path, self.args.data.molecule)
        if not self.dynamics:
            extra_args = {}
        else:
            extra_args = {'pred_step': self.args.task.pred_step}
        self.dataset = DatasetRegistry.get_dataset('md17' if not self.dynamics else 'md17_dynamics')(
            root=self.args.data.data_path, dataset_arg=self.args.data.molecule,
            transform=QM9_Transform(self.args.task.charge_power), **extra_args)
        print('Data ready')
        self.model_specific_args = {'node_dim': 5 * (self.args.task.charge_power + 1), 'edge_attr_dim': 0}
        self.trainer_specific_args = {'test': False}
        if self.args.trainer.exp_name is None:
            self.args.trainer.model_save_path = os.path.join(self.args.trainer.model_save_path,
                                                             self.args.model.name,
                                                             self.args.data.molecule)

    def task(self, encoder):
        task = Prediction(rep=encoder, rep_dim=self.args.model.hidden_dim, output_dim=1,
                          task_type='Regression', loss='MAE', decoding='MLP', vector_method='gradient',
                          normalize=(self.dataset.mean(), self.dataset.std()), scalar_pooling='sum',
                          target=['scalar', 'vector'],
                          loss_weight=[self.args.task.energy_weight, self.args.task.force_weight], return_outputs=False,
                          dynamics=self.dynamics)
        return task

    @property
    def trainer(self):
        return Trainer

    @property
    def dynamics(self):
        return False


@BenchmarkRegistry.register_benchmark('benchmark_md17_dynamics')
class BenchmarkMD17Dynamics(BenchmarkMD17):
    def __init__(self, args_file=None):
        super(BenchmarkMD17Dynamics, self).__init__(args_file)

        transform = MD17_Transform(max_atom_type=self.args.task.max_atom_type, charge_power=self.args.task.charge_power,
                                   atom_type_name='charge', cutoff=1.6, max_hop=self.args.task.max_hop)
        transform.get_example(self.dataset[0])
        self.dataset.transform = transform
        self.model_specific_args = {'node_dim': self.args.task.max_atom_type * (self.args.task.charge_power + 1),
                                    'edge_attr_dim': self.args.task.max_hop + 1}
        self.trainer_specific_args['rollout_step'] = self.args.task.rollout_step
        self.trainer_specific_args['save_pred'] = self.args.trainer.save_pred

    def task(self, encoder):
        task = Prediction(rep=encoder, output_dim=1, rep_dim=self.args.model.hidden_dim,
                          task_type='Regression', loss='MAE', decoding=None, vector_method='diff',
                          scalar_pooling=None, target='vector', dynamics=self.dynamics,
                          return_outputs=True)
        return task

    @property
    def trainer(self):
        return DynamicsTrainer

    @property
    def dynamics(self):
        return True


@BenchmarkRegistry.register_benchmark('benchmark_nbody_dynamics')
class BenchmarkNBodyDynamics(Benchmark):
    def __init__(self, args_file=None):
        if args_file is None:
            args_file = 'examples/configs/nbody_dynamics/egnn.yaml'
        super(BenchmarkNBodyDynamics, self).__init__(args_file)
        self.dataset = DatasetRegistry.get_dataset('nbody_dynamics')(
            root=self.args.data.data_path, transform=NBody_Transform, n_particle=self.args.data.n_particle,
            num_samples=self.args.data.num_samples, T=self.args.data.T, sample_freq=self.args.data.sample_freq,
            num_workers=20, initial_step=self.args.task.initial_step, pred_step=self.args.task.pred_step)
        self.model_specific_args = {'node_dim': 1, 'edge_attr_dim': 1}
        self.trainer_specific_args = {'test': True, 'save_pred': self.args.trainer.save_pred}

    def task(self, encoder):
        task = Prediction(rep=encoder, output_dim=1, rep_dim=self.args.model.hidden_dim,
                          task_type='Regression', loss='MSE', decoding=None, vector_method='diff',
                          scalar_pooling=None, target='vector', dynamics=self.dynamics,
                          return_outputs=True)
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


@BenchmarkRegistry.register_benchmark('benchmark_atom3d_lba')
class BenchmarkAtom3dLBA(Benchmark):
    def __init__(self, args_file=None):
        if args_file is None:
            args_file = 'examples/configs/atom3d_lba/egnn.yaml'
        super(BenchmarkAtom3dLBA, self).__init__(args_file)
        self.model_specific_args = {'node_dim': 61, 'edge_attr_dim': 1}

    def get_dataset_splits(self):
        transform = SelectEdges()
        dataset_fn = DatasetRegistry.get_dataset('atom3d')
        datasets = {_: dataset_fn(root=self.args.data.data_path, task='lba', split=_,
                                  dataset_arg='sequence-identity-30',
                                  transform=transform) for _ in ['train', 'val', 'test']}
        return datasets

    def task(self, encoder):
        task = Prediction(rep=encoder, output_dim=1, rep_dim=self.args.model.hidden_dim,
                          task_type='Regression', loss='MSE', decoding='MLP',
                          target='scalar', return_outputs=True, dynamics=self.dynamics)
        return task

    @property
    def dynamics(self):
        return False

    @property
    def trainer(self):
        return Trainer


@BenchmarkRegistry.register_benchmark('benchmark_atom3d_lep')
class BenchmarkAtom3dLEP(Benchmark):
    def __init__(self, args_file=None):
        if args_file is None:
            args_file = 'examples/configs/atom3d_lep/egnn.yaml'
        super(BenchmarkAtom3dLEP, self).__init__(args_file)
        self.model_specific_args = {'node_dim': 18, 'edge_attr_dim': 1}

    def get_dataset_splits(self):
        transform = LEP_Transform()
        dataset_fn = DatasetRegistry.get_dataset('atom3d')
        datasets = {_: dataset_fn(root=self.args.data.data_path, task='lep', split=_, dataset_arg='protein',
                                  transform=transform) for _ in ['train', 'val', 'test']}
        return datasets

    def task(self, encoder):
        task = Contrastive(rep=encoder, output_dim=1, rep_dim=self.args.model.hidden_dim,
                           task_type='BinaryClassification', loss='BCE',
                           return_outputs=True, dynamics=self.dynamics)
        return task

    @property
    def dynamics(self):
        return False

    @property
    def trainer(self):
        return Trainer
