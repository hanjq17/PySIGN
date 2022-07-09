from ..utils import get_default_args, load_params, set_seed
from ..nn.model import get_model_from_args
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
        self.model_specific_args = {}
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

    def get_dataset_splits(self):
        return self.dataset.default_split()

    def launch(self):
        self.datasets = self.get_dataset_splits()
        dataloaders = {
            split: DataLoader(self.datasets[split],
                              batch_size=self.args.batch_size, shuffle=True if split == 'train' else False)
            for split in self.datasets}
        encoder = get_model_from_args(args=self.args, dynamics=self.dynamics, **self.model_specific_args)
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
