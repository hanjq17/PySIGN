from .basic import Trainer


class DynamicsTrainer(Trainer):
    def __init__(self, dataloaders, task, args, device, lower_is_better=True, verbose=True, test=True):
        super(DynamicsTrainer, self).__init__(dataloaders, task, args, device, lower_is_better, verbose, test)

    def evaluate(self, valid=False, ):
        # TODO: add rollout prediction
        loader = self.dataloaders.get('val') if valid else self.dataloaders.get('test')
        all_loss = []
        for step, batch_data in enumerate(loader):
            batch_data = batch_data.to(self.device)
            _, loss = self.task(batch_data)
            all_loss.append(loss.detach().cpu().numpy())
        return self.stats.get_averaged_loss(all_loss)




