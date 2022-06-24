from torch_geometric.nn import global_mean_pool
import torch.nn as nn
from .prediction import Prediction
import torch


class Constrastive(Prediction):
    """
    The constrastive task.

    :param rep: the representation module.
    :param output_dim: the output dimension for computing loss.
    :param rep_dim: the dimension of the representation.
    """
    def __init__(self, rep, output_dim, rep_dim, task_type='Regression', loss='MSE'):
        self.activation = nn.ReLU()  # TODO: use args as activation
        super(Constrastive, self).__init__(rep, output_dim, rep_dim, task_type, loss)

    def get_decoder(self):
        """
        Instantiate the decoder module. Currently adopt a Linear layer.

        :return: The decoder module.
        """
        decoder = nn.Sequential(
            nn.Linear(self.rep_dim * 2, self.rep_dim * 2),
            self.activation,
            nn.Linear(self.rep_dim * 2, 1),
            nn.Sigmoid()
        )
        return decoder

    def forward(self, data):
        """
        Forward passing with the data object. First, the data is processed by the representation module.
        Afterwards, the representation is delivered to the decoder, and the output together with labels yield the loss.

        :param data: The data object.
        :return: The loss computed.
        """
        data1, data2 = data
        rep1, rep2 = self.rep(data1), self.rep(data2)
        rep1, rep2 = global_mean_pool(rep1.h, data1.batch), global_mean_pool(rep2.h, data2.batch)
        rep = torch.cat((rep1.h, rep2.h), dim=-1)
        pred = self.decoder(rep)
        loss = self.loss(pred, data1.y)
        return pred, loss, data1.y

