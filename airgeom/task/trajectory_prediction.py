from torch.nn import Linear, MSELoss, Module, L1Loss
from torch_geometric.nn import global_add_pool
from .basic import BasicTask
from .output_modules import *
from torch.autograd import grad


class TrajectoryPrediction(BasicTask):
    """
    The prediction task.

    :param rep: the representation module.
    :param output_dim: the output dimension for computing loss.
    :param rep_dim: the dimension of the representation.
    """
    def __init__(self, rep, rep_dim, decoder_type='DifferentialVector'):
        super(TrajectoryPrediction, self).__init__(rep)
        self.rep_dim = rep_dim
        assert decoder_type in ['Scalar','EquivariantScalar','DifferentialVector','EquivariantVector']
        self.decoder_type = decoder_type
        self.decoder = self.get_decoder()
        self.loss = self.get_loss()

    def get_decoder(self):
        """
        Instantiate the decoder module. Currently adopt a Linear layer.

        :return: The decoder module.
        """
        if self.decoder_type == 'Scalar':
            decoder = Scalar(self.rep_dim)
        elif self.decoder_type == 'EquivariantScalar':
            decoder = EquivariantScalar(self.rep_dim)
        elif self.decoder_type == 'EquivariantVector':
            decoder = EquivariantVector(self.rep_dim)
        elif self.decoder_type == 'DifferentialVector':
            decoder = DifferentialVector()
        return decoder

    def get_loss(self):
        """
        Instantiate the loss module. Currenly adopt the MSE loss.

        :return: The loss module.
        """
        loss = L1Loss(reduction='none')
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
        if self.decoder_type in ['Scalar', 'EquivariantScalar']:
            data.pos.requires_grad_(True)
        rep = self.rep(data)
        output = self.decoder(data)  # node-wise rep
        if self.decoder_type in ['Scalar', 'EquivariantScalar']:
            output = global_add_pool(output, data.batch)
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(output)]
            dy = - grad(
                [output],
                [data.pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            output = dy
        loss = self.loss(output, data.pred).sum(dim=-1)
        return output, loss



