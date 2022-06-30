from .basic import Trainer
import torch
import numpy as np
from copy import deepcopy
from torch_geometric.nn import global_mean_pool


class DynamicsTrainer(Trainer):
    def __init__(self, dataloaders, task, args, device, lower_is_better=True, verbose=True, test=True, rollout_step=1, save_pred=False):
        super(DynamicsTrainer, self).__init__(dataloaders, task, args, device, lower_is_better, verbose, test)
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
        x_pred, v_pred, x_true = None, None, None
        for step, batch_data in enumerate(loader):
            batch_data = batch_data.to(self.device)
            if step % self.rollout_step == 0:
                x_true = batch_data.x + batch_data.v_label  # pred is v_label
            else:
                x_true = x_true + batch_data.v_label
            if step % self.rollout_step > 0:
                batch_data.v = v_pred
                batch_data.x = x_pred
            v_pred, loss, v_label = self.task(batch_data)
            x_pred = (batch_data.x + v_pred).detach()
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

    def evaluate_rollout_multi_system(self, valid=False):
        loader = self.dataloaders.get('val') if valid else self.dataloaders.get('test')
        assert loader.dataset.mode == 'rollout'
        rollout_step = loader.dataset.rollout_step
        all_graph_loss = []
        all_pred = []
        for step, batch_data in enumerate(loader):
            batch_data = batch_data.to(self.device)
            _v_label = deepcopy(batch_data.v_label)
            x_pred, v_pred, x_true = None, None, None
            cur_all_graph_loss = []
            cur_pred = []
            for t in range(rollout_step):
                if t == 0:
                    batch_data.x = batch_data.x[:, t, :]
                    batch_data.v_label = _v_label[:, t, :]
                    batch_data.v = batch_data.v[:, t, :]
                    x_true = batch_data.x + batch_data.v_label
                else:
                    batch_data.x = x_pred
                    batch_data.v = v_pred
                    batch_data.v_label = _v_label[:, t, :]
                    x_true = x_true + batch_data.v_label

                cur_pred.append(batch_data.x)
                
                # TODO: revise here, not a good implementation
                batch_data.h = torch.norm(batch_data.v, dim=-1, keepdim=True)
                v_pred, loss, v_label = self.task(batch_data)
                x_pred = (batch_data.x + v_pred).detach()
                v_pred = v_pred.detach()
                cur_rollout_loss = self.mse_loss(x_pred, x_true).mean(dim=-1, keepdim=True)  # [BN, 1]
                cur_graph_rollout_loss = global_mean_pool(cur_rollout_loss, batch_data.batch)  # [BG, 1]
                cur_all_graph_loss.append(cur_graph_rollout_loss)
            cur_all_graph_loss = torch.cat(cur_all_graph_loss, dim=-1)  # [BG, T]
            all_graph_loss.append(cur_all_graph_loss)
            cur_pred = torch.stack(cur_pred, dim=1)  # [BN, T, 3]
            batch_data.x = cur_pred
            split_data = batch_data.to_data_list()
            all_pred.extend([_.x.transpose(0, 1).detach().cpu().numpy() for _ in split_data])
        all_graph_loss = torch.cat(all_graph_loss, dim=0).detach().cpu().numpy()  # [Tot_G, T]
        if self.save_pred:
            return all_graph_loss, all_pred  # [Tot_G, T], [Tot_G, T, N, 3]
        else:
            return all_graph_loss, None








