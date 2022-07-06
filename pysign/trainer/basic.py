from ..utils import get_optimizer, get_scheduler, StatsCollector, EarlyStopping, SavingHandler


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

    def train_epoch(self):
        train_loader = self.dataloaders.get('train')
        for step, batch_data in enumerate(train_loader):
            if isinstance(batch_data, list):
                batch_data = [_.to(self.device) for _ in batch_data]
            else:
                batch_data = batch_data.to(self.device)
            self.optimizer.zero_grad()
            _, loss, y = self.task(batch_data)
            self.stats.update_step({'train_loss': loss})
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            if self.verbose:
                if step % 40 == 0:
                    print('Step: {:4d} | Train Loss: {:.6f}'.format(step, loss.item()))

    def evaluate(self, valid=False):
        loader = self.dataloaders.get('val') if valid else self.dataloaders.get('test')
        all_loss = []
        for step, batch_data in enumerate(loader):
            batch_data = batch_data.to(self.device)
            _, loss, y = self.task(batch_data)
            all_loss.append(loss.detach().cpu().numpy())
        return {'loss': self.stats.get_averaged_loss(all_loss)}

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




