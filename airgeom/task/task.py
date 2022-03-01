from torch.nn import Linear, MSELoss, Module
from torch_geometric.nn import global_mean_pool


class BaseTask(Module):
    def __init__(self, rep):
        super(BaseTask, self).__init__()
        self.rep = rep

    def get_decoder(self):
        raise NotImplementedError()

    def get_loss(self):
        raise NotImplementedError()

    @property
    def params(self):
        raise NotImplementedError()

    def __call__(self, data):
        raise NotImplementedError()


class Prediction(BaseTask):
    def __init__(self, rep, output_dim, rep_dim):
        super(Prediction, self).__init__(rep)
        self.rep_dim = rep_dim
        self.output_dim = output_dim
        self.decoder = self.get_decoder()
        self.loss = self.get_loss()

    def get_decoder(self):
        decoder = Linear(self.rep_dim, self.output_dim)
        return decoder

    def get_loss(self):
        loss = MSELoss(reduction='none')
        return loss

    @property
    def params(self):
        return self.decoder.parameters()

    def __call__(self, data):
        rep = self.rep(data)
        output = self.decoder(rep.h)  # node-wise rep
        output = global_mean_pool(output, data.batch)
        loss = self.loss(output.squeeze(-1), data.y.squeeze(-1))
        return output, loss



