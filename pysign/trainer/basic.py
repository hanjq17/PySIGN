from ..utils import get_optimizer, get_scheduler, StatsCollector, EarlyStopping, SavingHandler
import numpy as np
import scipy.stats
from sklearn.metrics import roc_auc_score, average_precision_score
import torch


class Trainer(object):
    def __init__(self, dataloaders, task, args, device, lower_is_better=True, verbose=True, test=True):
        self.dataloaders: dict = dataloaders
        self.task = task.to(device)
        self.args = args
        self.device = device
        self.verbose = verbose
        self.test = test
        self.optimizer = get_optimizer(args.optimizer, args.lr, args.weight_decay, task.params)
        self.scheduler = get_scheduler(args.scheduler, self.optimizer, args)
        self.stats = StatsCollector()
        self.early_stopping = EarlyStopping(lower_is_better=lower_is_better, max_times=args.earlystopping,
                                            verbose=verbose)
        self.model_saver = SavingHandler(self.task, save_path=args.model_save_path,
                                         lower_is_better=lower_is_better, max_instances=5)

        self.full_eval_metrics_flag = ('scalar' in self.task.target) and self.task.return_outputs

    def train_epoch(self):
        train_loader = self.dataloaders.get('train')
        for step, batch_data in enumerate(train_loader):
            if isinstance(batch_data, list):
                batch_data = [_.to(self.device) for _ in batch_data]
            else:
                batch_data = batch_data.to(self.device)
            self.optimizer.zero_grad()
            tot_loss, all_loss, outputs = self.task(batch_data)
            self.stats.update_step({'train_loss': tot_loss})
            tot_loss = tot_loss.mean()
            tot_loss.backward()
            self.optimizer.step()
            if self.verbose:
                if step % 40 == 0:
                    format_str = ''
                    if len(all_loss) > 1:  # The multi-task scenario
                        for k, v in all_loss.items():
                            if k.startswith('loss_'):
                                cur_loss_name = k[5:]
                                format_str += f' | {cur_loss_name}: {v.mean().item()}'
                    print('Step: {:4d} | Train Loss: {:.6f}'.format(step, tot_loss.item()) + format_str)

    def evaluate(self, valid=False):
        loader = self.dataloaders.get('val') if valid else self.dataloaders.get('test')
        loss_recorder = {'loss': []}
        output_recorder = []

        for step, batch_data in enumerate(loader):

            if isinstance(batch_data, list):
                batch_data = [_.to(self.device) for _ in batch_data]
            else:
                batch_data = batch_data.to(self.device)

            tot_loss, all_loss, outputs = self.task(batch_data)
            loss_recorder['loss'].append(tot_loss.detach().cpu().numpy())

            if len(all_loss) > 1:  # The multi-task scenario
                for k, v in all_loss.items():
                    if k.startswith('loss_'):
                        cur_loss_name = k[5:]
                        if cur_loss_name in loss_recorder:
                            loss_recorder[cur_loss_name].append(v.detach().cpu().numpy())
                        else:
                            loss_recorder[cur_loss_name] = [v.detach().cpu().numpy()]

            if self.full_eval_metrics_flag:
                # TODO: should this extend to multitask scenario?
                pred, y = outputs['scalar']
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
                output_recorder.append(res)

        metrics = {k: self.stats.get_averaged_loss(v) for k, v in loss_recorder.items()}
        # Compute full evaluation metrics
        if self.full_eval_metrics_flag:
            if self.task.task_type == 'Regression':
                mae = self.stats.get_averaged_loss([_[0] for _ in output_recorder])
                metrics['mae'] = mae
                mse = self.stats.get_averaged_loss([_[1] for _ in output_recorder])
                metrics['mse'] = mse
                metrics['rmse'] = np.sqrt(mse)
                y_pred = np.concatenate([_[2] for _ in output_recorder], axis=0)
                y_true = np.concatenate([_[3] for _ in output_recorder], axis=0)
                pearson = np.corrcoef(y_true, y_pred)[0, 1]
                spearmanr = scipy.stats.spearmanr(y_true, y_pred)[0]
                metrics['pearson'] = pearson
                metrics['spearmanr'] = spearmanr
            elif self.task.task_type == 'BinaryClassification':
                acc = self.stats.get_averaged_loss([_[0] for _ in output_recorder])
                metrics['acc'] = acc
                y_pred = np.concatenate([_[1] for _ in output_recorder], axis=0)
                y_true = np.concatenate([_[2] for _ in output_recorder], axis=0)
                auroc = roc_auc_score(y_true, y_pred)
                auprc = average_precision_score(y_true, y_pred)
                metrics['auroc'] = auroc
                metrics['auprc'] = auprc
            elif self.task.task_type == 'MultiClassification':
                acc = self.stats.get_averaged_loss([_[0] for _ in output_recorder])
                metrics['acc'] = acc
            else:
                raise NotImplementedError('Not implement task type:', self.task.task_type)

        return metrics

    def loop(self):
        for ep in range(self.args.epoch):
            print('Starting epoch ', ep)
            self.train_epoch()
            train_loss = self.stats.get_train_loss()

            if ep % self.args.eval_epoch == 0:
                val_result = self.evaluate(valid=True)
                val_loss = val_result.get('loss')
                if self.scheduler is not None:
                    self.scheduler.step(metrics=val_loss)
                better = self.early_stopping(val_loss)
                if better == 'exit':
                    return
                if better:
                    self.model_saver(ep, val_loss)
                if self.test:
                    test_result = self.evaluate(valid=False)
                else:
                    test_result = None
            else:
                val_result, test_result = None, None
            stats = {'train_loss': train_loss}
            if val_result is not None:
                for metric in val_result:
                    stats['val_' + metric] = val_result[metric]
            if test_result is not None:
                for metric in test_result:
                    stats['test_' + metric] = test_result[metric]
            self.stats.update_epoch(stats)
            print('Epoch: {:4d} | LR: {:.6f} | Train Loss: {:.6f}'.
                  format(ep, self.optimizer.param_groups[0]['lr'], train_loss))
            if val_result is not None:
                print('Val ', end=' ')
                for metric in val_result:
                    print('{}: {:.6f}'.format(metric, val_result[metric]), end=' ')
                print()
            if test_result is not None:
                print('Test', end=' ')
                for metric in test_result:
                    print('{}: {:.6f}'.format(metric, test_result[metric]), end=' ')
                print()

        print('Finished!')




