from .basic import Trainer
import torch
import numpy as np


class DynamicsTrainer(Trainer):
    def __init__(self, dataloaders, task, args, device, lower_is_better=True, verbose=True, test=True, rollout_step=1, save_pred=False):
        super(DynamicsTrainer, self).__init__(dataloaders, task, args, device, lower_is_better, verbose, test)
        self.rollout_step = rollout_step
        self.save_pred = save_pred
        self.mae_loss = torch.nn.L1Loss()

    def evaluate_rollout(self, valid=False):
        loader = self.dataloaders.get('val') if valid else self.dataloaders.get('test')
        all_loss = []
        cur_loss = []
        all_pred = []
        cur_pred = []
        x_pred, v_pred = None, None
        for step, batch_data in enumerate(loader):
            batch_data = batch_data.to(self.device)
            x_true = batch_data.pos + batch_data.pred
            if step % self.rollout_step > 0:
                batch_data.v = v_pred
                batch_data.pos = x_pred
            v_pred, loss = self.task(batch_data)
            x_pred = (batch_data.pos + v_pred).detach()
            v_pred = v_pred.detach()
            if self.save_pred:
                cur_pred.append(x_pred.detach().cpu().numpy())
            cur_rollout_loss = self.mae_loss(x_pred, x_true)
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




