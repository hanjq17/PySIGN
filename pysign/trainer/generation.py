from .basic import Trainer
import torch
import numpy as np
from copy import deepcopy
from torch_geometric.nn import global_mean_pool
from pysign.utils.align import kabsch


class ConformationTrainer(Trainer):
    def __init__(self, dataloaders, task, args, device, lower_is_better=True, verbose=True, test=True, rollout_step=1, save_pred=False):
        super(ConformationTrainer, self).__init__(dataloaders, task, args, device, lower_is_better, verbose, test)
        self.rollout_step = rollout_step
        self.save_pred = save_pred
        self.mae_loss = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss(reduction='none')

    def evaluate_rollout(self, valid=False):
        loader = self.dataloaders.get('val') if valid else self.dataloaders.get('test')
        all_loss = []
        cur_loss = []
        all_pred = []
        cur_pred = []
        for step, batch_data in enumerate(loader):
            batch_data = batch_data.to(self.device)
            x_true = batch_data.x
            x_pred = self.task.generate(batch_data)
            batch_data.x = torch.cat([x_true, x_pred], dim=-1)
            data_list = batch_data.to_data_list()
            cur_rollout_loss = []
            for item in data_list:
                dim = item.x.shape[-1] // 2
                item_x_pred = item.x[..., dim:].detach().cpu().numpy()
                item_x_true = item.x[..., :dim].detach().cpu().numpy()
                item_x_pred, _, _ = kabsch(item_x_pred, item_x_true)
                if self.save_pred:
                    cur_pred.append(item_x_pred)
                cur_rollout_loss.append(self.mse_loss(torch.tensor(item_x_pred), torch.tensor(item_x_true)).mean())
            cur_rollout_loss = torch.mean(torch.tensor(cur_rollout_loss))
            cur_loss.append(cur_rollout_loss.item())
            if step % 40 == 0:
                print('Step: {:4d} | Cur Loss: {:.6f}'.format(step, cur_rollout_loss.item()))
            if step % self.rollout_step == self.rollout_step - 1:
                all_loss.append(cur_loss)
                cur_loss = []
                if self.save_pred:
                    all_pred.append(cur_pred)
                    cur_pred = []
                print('***' * 10)
        all_loss = np.array(all_loss)
        if self.save_pred:
            all_pred = np.array(all_pred)
            return all_loss, all_pred  # [n_traj, traj_len, N, 3]
        else:
            return all_loss, None  # [n_traj, traj_len]