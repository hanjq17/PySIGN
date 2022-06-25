from torch.nn import Linear, MSELoss, Module, L1Loss, NLLLoss, BCELoss, CrossEntropyLoss
from torch_geometric.nn import global_mean_pool
from .basic import BasicTask


class Prediction(BasicTask):
    """
    The prediction task.

    :param rep: the representation module.
    :param output_dim: the output dimension for computing loss.
    :param rep_dim: the dimension of the representation.
    """
    def __init__(self, rep, output_dim, rep_dim, task_type='Regression', loss='MSE', mean=None, std=None):
        super(Prediction, self).__init__(rep)
        self.rep_dim = rep_dim
        self.output_dim = output_dim
        self.task_type = task_type
        self.decoder = self.get_decoder()
        self.loss = self.get_loss(loss)
        self.mean = mean
        self.std = std

    def get_decoder(self):
        """
        Instantiate the decoder module. Currently adopt a Linear layer.

        :return: The decoder module.
        """
        decoder = Linear(self.rep_dim, self.output_dim)
        return decoder

    def get_loss(self, loss):
        """
        Instantiate the loss module. Currenly adopt the MSE loss.

        :return: The loss module.
        """
        if loss == 'MAE':
            loss = L1Loss(reduction='none')
            assert self.task_type == 'Regression'
        elif loss == 'MSE':
            loss = MSELoss(reduction='none')
            assert self.task_type == 'Regression'
        elif loss == 'BCE':
            loss = BCELoss(reduction='none')
            assert self.task_type == 'BinaryClassification'
        elif loss == 'CE':
            loss = CrossEntropyLoss(reduction='none')
            assert self.task_type == 'MultiClassification'
        else:
            raise NotImplementedError('Unknown loss type', loss)
        return loss

    @property
    def params(self):
        """
        Get the parameters to optimize.

        :return: The parameters to optimize.
        """
        return self.parameters()

    def forward(self, data):
        """
        Forward passing with the data object. First, the data is processed by the representation module.
        Afterwards, the representation is delivered to the decoder, and the output together with labels yield the loss.

        :param data: The data object.
        :return: The loss computed.
        """
        rep = self.rep(data)
        output = self.decoder(rep.h)  # node-wise rep
        output = global_mean_pool(output, data.batch)
        if len(output.shape) == 2:
            output = output.squeeze(-1)
        if len(data.y.shape) == 2:
            y = data.y.squeeze(-1)
        else:
            y = data.y
        if self.std is not None:
            output = output * self.std
        if self.mean is not None:
            output = output + self.mean
        loss = self.loss(output, y)
        return output, loss, y



