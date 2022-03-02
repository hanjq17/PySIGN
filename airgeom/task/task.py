from torch.nn import Linear, MSELoss, Module
from torch_geometric.nn import global_mean_pool


class BaseTask(Module):
    """
    The basic task.

    :param rep: the representation module.
    """
    def __init__(self, rep):
        super(BaseTask, self).__init__()
        self.rep = rep

    def get_decoder(self):
        """
        Instantiate the decoder module.

        :return: The decoder module.
        """
        raise NotImplementedError()

    def get_loss(self):
        """
        Instantiate the loss module.

        :return: The loss module.
        """
        raise NotImplementedError()

    @property
    def params(self):
        """
        Get the parameters to optimize.

        :return: The parameters to optimize.
        """
        raise NotImplementedError()

    def __call__(self, data):
        """
        Forward passing with the data object.

        :param data: The data object.
        :return: The loss computed.
        """
        raise NotImplementedError()


class Prediction(BaseTask):
    """
    The prediction task.

    :param rep: the representation module.
    :param output_dim: the output dimension for computing loss.
    :param rep_dim: the dimension of the representation.
    """
    def __init__(self, rep, output_dim, rep_dim):
        super(Prediction, self).__init__(rep)
        self.rep_dim = rep_dim
        self.output_dim = output_dim
        self.decoder = self.get_decoder()
        self.loss = self.get_loss()

    def get_decoder(self):
        """
        Instantiate the decoder module. Currently adopt a Linear layer.

        :return: The decoder module.
        """
        decoder = Linear(self.rep_dim, self.output_dim)
        return decoder

    def get_loss(self):
        """
        Instantiate the loss module. Currenly adopt the MSE loss.

        :return: The loss module.
        """
        loss = MSELoss(reduction='none')
        return loss

    @property
    def params(self):
        """
        Get the parameters to optimize.

        :return: The parameters to optimize.
        """
        return self.parameters()

    def __call__(self, data):
        """
        Forward passing with the data object. First, the data is processed by the representation module.
        Afterwards, the representation is delivered to the decoder, and the output together with labels yield the loss.

        :param data: The data object.
        :return: The loss computed.
        """
        rep = self.rep(data)
        output = self.decoder(rep.h)  # node-wise rep
        output = global_mean_pool(output, data.batch)
        loss = self.loss(output.squeeze(-1), data.y.squeeze(-1))
        return output, loss



