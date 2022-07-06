from torch.nn import Linear, MSELoss, L1Loss, BCELoss, CrossEntropyLoss
from torch_geometric.nn import global_mean_pool, global_add_pool
from .basic import BasicTask
from .utils.output_modules import *
from torch.autograd import grad
import torch


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
        Instantiate the loss module. Currently adopt the MSE loss.

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
        output = self.decoder(rep.h_pred)  # node-wise rep
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


class EnergyForcePrediction(BasicTask):
    """
    The energy & force prediction task.

    :param rep: the representation module.
    :param output_dim: the output dimension for computing loss.
    :param rep_dim: the dimension of the representation.
    """
    def __init__(self, rep, rep_dim, decoder_type, task_type='Regression', loss='MSE', mean=None, std=None, energy_weight=0.2, force_weight=0.8):
        super(EnergyForcePrediction, self).__init__(rep)
        self.rep_dim = rep_dim
        self.task_type = task_type
        assert decoder_type in ['Scalar', 'DifferentialVector', 'EquivariantVector']
        self.decoder_type = decoder_type
        self.decoder = self.get_decoder()
        self.loss = self.get_loss(loss)
        self.mean = mean
        self.std = std
        self.energy_weight = energy_weight
        self.force_weight = force_weight

    def get_decoder(self):
        """
        Instantiate the decoder module. Currently adopt a Linear layer.

        :return: The decoder module.
        """
        if self.decoder_type == 'Scalar':
            decoder = Scalar(self.rep_dim)
        elif self.decoder_type == 'EquivariantVector':
            decoder = EquivariantVector(self.rep_dim)
        elif self.decoder_type == 'DifferentialVector':
            decoder = Scalar(self.rep_dim)
            if not hasattr(self, 'decoder_f'):
                self.decoder_f = DifferentialVector()
        else:
            raise NotImplementedError('Unknown decoder type:', self.decoder_type)
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
        if self.decoder_type in ['Scalar']:
            data.x.requires_grad_(True)
        rep = self.rep(data)
        output = self.decoder(rep)  # node-wise rep
        if self.decoder_type == 'DifferentialVector':
            energy = global_add_pool(output, data.batch)
            if self.std is not None:
                energy = energy * self.std
            if self.mean is not None:
                energy = energy + self.mean
            force = self.decoder_f(rep)
        elif self.decoder_type == 'Scalar':
            energy = global_add_pool(output, data.batch)
            if self.std is not None:
                energy = energy * self.std
            if self.mean is not None:
                energy = energy + self.mean
            grad_outputs = [torch.ones_like(energy)]
            force = - grad(
                [energy],
                [data.x],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
        elif self.decoder_type == 'EquivariantVector':
            energy, force = output
            energy = global_add_pool(energy, data.batch)
            if self.std is not None:
                energy = energy * self.std
            if self.mean is not None:
                energy = energy + self.mean
        else:
            raise NotImplementedError('Unknown decoder type:', self.decoder_type) 
        energy = energy.reshape(-1)
        target = data.y.reshape(-1)
        loss_energy = self.loss(energy, target)
        loss_force = self.loss(force, data.dy).mean(dim=-1)
        loss_force = global_mean_pool(loss_force, data.batch)
        loss = self.energy_weight * loss_energy + self.force_weight * loss_force
        return loss, {
            'loss_energy': loss_energy,
            'loss_force': loss_force
        }



