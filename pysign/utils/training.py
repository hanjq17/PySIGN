from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import torch
import random


def get_optimizer(opt_args, params):
    if opt_args.name == 'SGD':
        return SGD(lr=opt_args.lr, weight_decay=opt_args.weight_decay, params=params)
    elif opt_args.name == 'Adam':
        return Adam(lr=opt_args.lr, weight_decay=opt_args.weight_decay, params=params)
    else:
        raise NotImplementedError('Unknown optimizer', opt_args.name)


def get_scheduler(scheduler_args, opt):
    if scheduler_args.name == 'Plateau':
        return ReduceLROnPlateau(optimizer=opt, factor=scheduler_args.factor,
                                 patience=scheduler_args.patience, verbose=True)
    else:
        raise NotImplementedError('Unknown scheduler', scheduler_args.name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print('Set torch and numpy seeds to', seed)


def to_numpy(stats):
    # convert to numpy
    for k in stats:
        try:
            stats[k] = stats[k].detach().cpu().numpy()
        except:
            pass
    return stats


class StatsCollector(object):
    def __init__(self):
        self.epoch = 0
        self.step = 0
        self.stats = {}

    def update_step(self, stats):
        # convert to numpy
        stats = to_numpy(stats)
        if self.epoch not in self.stats:
            self.stats[self.epoch] = {}
        self.stats[self.epoch][self.step] = stats
        self.step += 1

    def update_epoch(self, stats):
        # convert to numpy
        stats = to_numpy(stats)
        self.stats[self.epoch]['info'] = stats
        self.step = 0
        self.epoch += 1

    def get_train_loss(self):
        all_train_loss = [self.stats[self.epoch][d]['train_loss'] for d in self.stats[self.epoch] if d != 'info']
        return self.get_averaged_loss(all_train_loss)

    @staticmethod
    def get_averaged_loss(all_loss):
        loss = np.concatenate(all_loss, axis=0).mean()
        return loss


class SavingHandler(object):
    def __init__(self, model, save_path, lower_is_better=True, max_instances=5):
        self.lower_is_better = lower_is_better
        self.max_instances = max_instances
        self.model = model
        self.save_path = save_path
        self.saved_models = []

    def __call__(self, epoch, metric):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if len(self.saved_models) >= self.max_instances:
            # pop the worst model
            path_to_rm = os.path.join(self.save_path, 'epoch_' + str(self.saved_models[0][0]) + '.pth')
            try:
                os.remove(path_to_rm)
            except FileNotFoundError:
                print('Model checkpoint has already been removed', path_to_rm)
            self.saved_models = self.saved_models[1:]
        self.saved_models.append((epoch, metric))
        save_name = os.path.join(self.save_path, 'epoch_' + str(epoch) + '.pth')
        torch.save(self.model.state_dict(), save_name)

    def load(self, epoch='best'):
        models = os.listdir(self.save_path)
        models = [_ for _ in models if 'epoch' in _]
        epoch_idx = [int(_.split('_')[-1].split('.')[0]) for _ in models]
        idx = np.argmax(epoch_idx)
        best_model = models[idx]
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, best_model)))
        print('Loaded from', os.path.join(self.save_path, best_model))


class EarlyStopping(object):
    def __init__(self, lower_is_better=True, max_times=10, verbose=True):
        self.lower_is_better = lower_is_better
        self.max_times = max_times
        self.cur_best = np.inf if self.lower_is_better else - np.inf
        self.counter = 0
        self.verbose = verbose

    def __call__(self, metric):
        better = metric < self.cur_best if self.lower_is_better else metric > self.cur_best
        if better:
            self.counter = 0
            self.cur_best = metric
        else:
            self.counter += 1
            if self.verbose:
                print('Early Stopping counter:', self.counter)
            if self.counter > self.max_times:
                print('Early Stopping with patience', self.max_times, 'epochs, exit!')
                return 'exit'
        return better
