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
            batch_data = batch_data.to(self.device)
            self.optimizer.zero_grad()
            _, loss = self.task(batch_data)
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
            _, loss = self.task(batch_data)
            all_loss.append(loss.detach().cpu().numpy())
        return self.stats.get_averaged_loss(all_loss)

    def loop(self):
        for ep in range(self.args.epoch):
            print('Starting epoch ', ep)
            self.train_epoch()
            train_loss = self.stats.get_train_loss()

            if ep % self.args.eval_epoch == 0:
                val_loss = self.evaluate(valid=True)
                if self.scheduler is not None:
                    self.scheduler.step(metrics=val_loss)
                better = self.early_stopping(val_loss)
                if better:
                    self.model_saver(ep, val_loss)
                if self.test:
                    test_loss = self.evaluate(valid=False)
                else:
                    test_loss = 0.0
            else:
                val_loss, test_loss = None, None
            stats = {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}
            self.stats.update_epoch(stats)
            print('Epoch: {:4d} | LR: {:.6f} | Train Loss: {:.6f} | '
                  'Val Loss {:.6f} | Test Loss {:.6f}'.format(ep, self.optimizer.param_groups[0]['lr'],
                                                              train_loss, val_loss, test_loss))

        print('Finished!')




