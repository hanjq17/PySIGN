from .basic import Trainer
import torch
import numpy as np
import scipy.stats
from sklearn.metrics import roc_auc_score, average_precision_score


class PredictionTrainer(Trainer):
    def __init__(self, dataloaders, task, args, device, lower_is_better=True, verbose=True, test=True):
        super(PredictionTrainer, self).__init__(dataloaders, task, args, device, lower_is_better, verbose, test)

    def evaluate(self, valid=False):
        loader = self.dataloaders.get('val') if valid else self.dataloaders.get('test')
        all_loss = []
        all_res = []
        for step, batch_data in enumerate(loader):
            batch_data = batch_data.to(self.device)
            pred, loss = self.task(batch_data)
            if len(pred) == 2:
                pred = pred.squeeze(-1)
            y = batch_data.y
            if len(y) == 2:
                y = y.squeeze(-1)
            all_loss.append(loss.detach().cpu().numpy())

            if self.task.task_type == 'Regression':
                res = [torch.nn.L1Loss(reduction='none')(pred, y).detach().cpu().numpy(),
                       torch.nn.MSELoss(reduction='none')(pred, y).detach().cpu().numpy(),
                       pred.detach().cpu().numpy(),
                       y.detach().cpu().numpy()]
            elif self.task.task_type == 'BinaryClassification':
                temp = torch.sigmoid(pred)
                res = [((temp > 0.5).long() == y).float().detach().cpu().numpy(),
                       temp.detach().cpu().numpy(),
                       y.detach().cpu().numpy()]
            elif self.task.task_type == 'MultiClassification':
                res = [(torch.argmax(pred).long() == y).float().detach().cpu().numpy()]
            else:
                raise NotImplementedError('Not implement task type:', self.task.task_type)
            all_res.append(res)

        metrics = {}
        if self.task.task_type == 'Regression':
            mae = self.stats.get_averaged_loss([_[0] for _ in all_res])
            metrics['mae'] = mae
            mse = self.stats.get_averaged_loss([_[1] for _ in all_res])
            metrics['mse'] = mse
            metrics['rmse'] = np.sqrt(mse)
            y_pred = np.concatenate([_[2] for _ in all_res], axis=0)
            y_true = np.concatenate([_[3] for _ in all_res], axis=0)
            pearson = np.corrcoef(y_true, y_pred)[0, 1]
            spearmanr = scipy.stats.spearmanr(y_true, y_pred)[0]
            metrics['pearson'] = pearson
            metrics['spearmanr'] = spearmanr
        elif self.task.task_type == 'BinaryClassification':
            acc = self.stats.get_averaged_loss([_[0] for _ in all_res])
            metrics['acc'] = acc
            y_pred = np.concatenate([_[1] for _ in all_res], axis=0)
            y_true = np.concatenate([_[2] for _ in all_res], axis=0)
            auroc = roc_auc_score(y_true, y_pred)
            auprc = average_precision_score(y_true, y_pred)
            metrics['auroc'] = auroc
            metrics['auprc'] = auprc
        elif self.task.task_type == 'MultiClassification':
            acc = self.stats.get_averaged_loss([_[0] for _ in all_res])
            metrics['acc'] = acc
        else:
            raise NotImplementedError('Not implement task type:', self.task.task_type)

        metrics['loss'] = self.stats.get_averaged_loss(all_loss)
        return metrics






