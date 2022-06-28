from .basic import Trainer


class MultiTaskTrainer(Trainer):

    def train_epoch(self):
        train_loader = self.dataloaders.get('train')
        for step, batch_data in enumerate(train_loader):
            if isinstance(batch_data, list):
                batch_data = [_.to(self.device) for _ in batch_data]
            else:
                batch_data = batch_data.to(self.device)
            self.optimizer.zero_grad()
            loss, outputs = self.task(batch_data)
            self.stats.update_step({'train_loss': loss})
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            if self.verbose:
                if step % 40 == 0:
                    format_str = ''
                    for k, v in outputs.items():
                        if k.startswith('loss_'):
                            format_str += f' | {k[5:]}: {v.mean().item()}'
                    print('Step: {:4d} | Train Loss: {:.6f}'.format(step, loss.item()) + format_str)

    def evaluate(self, valid=False):
        loader = self.dataloaders.get('val') if valid else self.dataloaders.get('test')
        all_loss = {'loss':[]}
        for step, batch_data in enumerate(loader):
            batch_data = batch_data.to(self.device)
            loss, outputs = self.task(batch_data)
            all_loss['loss'].append(loss.detach().cpu().numpy())
            for k,v in outputs.items():
                if k.startswith('loss_'):
                    if k[5:] in all_loss:
                        all_loss[k[5:]].append(v.detach().cpu().numpy())
                    else:
                        all_loss[k[5:]] = [v.detach().cpu().numpy()]
        all_loss = {k:self.stats.get_averaged_loss(v) for k, v in all_loss.items()}
        return all_loss

