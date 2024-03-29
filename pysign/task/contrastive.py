from torch_geometric.nn import global_mean_pool
import torch.nn as nn
from .prediction import Prediction
import torch


class Contrastive(Prediction):
    """
    The constrastive task.

    :param rep: the representation module.
    :param output_dim: the output dimension for computing loss.
    :param rep_dim: the dimension of the representation.
    """

    def __init__(self, rep, output_dim, rep_dim, task_type='Regression', loss='MSE',
                 return_outputs=False, dynamics=False):
        super(Contrastive, self).__init__(rep, output_dim, rep_dim, task_type, loss,
                                          return_outputs=return_outputs, dynamics=dynamics)
        # self.activation = nn.ReLU()  # TODO: use args as activation

    def get_decoder(self):
        """
        Instantiate the decoder module. Currently adopt a Linear layer.

        :return: The decoder module.
        """
        decoder = nn.Sequential(
            nn.Linear(self.rep_dim * 2, self.rep_dim * 2),
            nn.ReLU(),
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
        rep1, rep2 = global_mean_pool(rep1.h_pred, data1.batch), global_mean_pool(rep2.h_pred, data2.batch)
        rep = torch.cat((rep1, rep2), dim=-1)
        pred = self.decoder(rep)
        pred = pred.squeeze(-1)
        y = data1.y
        loss = self.loss(pred, y)
        if self.return_outputs:
            outputs = {'scalar': (pred, y)}
        else:
            outputs = {}
        return loss, {'scalar': loss}, outputs
