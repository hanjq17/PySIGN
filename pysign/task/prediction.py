from torch.nn import MSELoss, L1Loss, BCELoss, CrossEntropyLoss
from .basic import BasicTask
from ..nn.model import GeneralPurposeDecoder
from torch_geometric.nn import global_mean_pool, global_add_pool


class Prediction(BasicTask):
    """
    The prediction task.

    :param rep: the representation module.
    :param output_dim: the output dimension for computing loss.
    :param rep_dim: the dimension of the representation.
    """
    def __init__(self, rep, output_dim, rep_dim, task_type='Regression', loss='MSE',
                 decoding='MLP', vector_method=None, normalize=None, scalar_pooling='sum', target='scalar',
                 loss_weight=None, return_outputs=False, dynamics=False):
        super(Prediction, self).__init__(rep)
        self.rep_dim = rep_dim
        self.output_dim = output_dim
        self.task_type = task_type
        self.loss = self.get_loss(loss)
        self.decoding = decoding
        self.vector_method = vector_method
        self.normalize = normalize
        self.scalar_pooling = scalar_pooling
        if not isinstance(target, list):
            target = [target]
        if loss_weight is None or not isinstance(loss_weight, list):
            loss_weight = [1.0]
        self.target = target
        self.loss_weight = loss_weight
        self.return_outputs = return_outputs
        self.dynamics = dynamics
        if dynamics:
            assert self.task_type == 'Regression'
        self.decoder = self.get_decoder()

    def get_decoder(self):
        """
        Instantiate the decoder module. Currently adopt a Linear layer.

        :return: The decoder module.
        """
        decoder = GeneralPurposeDecoder(hidden_dim=self.rep_dim, output_dim=self.output_dim, decoding=self.decoding,
                                        vector_method=self.vector_method, normalize=self.normalize,
                                        dynamics=self.dynamics)
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
        if self.vector_method == 'gradient':
            data.x.requires_grad_(True)

        rep = self.rep(data)
        scalar, vector = self.decoder(rep)
        if self.scalar_pooling == 'mean':
            scalar = global_mean_pool(scalar, data.batch)
        elif self.scalar_pooling == 'sum':
            scalar = global_add_pool(scalar, data.batch)
        else:
            pass
        if self.normalize is not None:
            scalar = scalar * self.normalize[1] + self.normalize[0]

        all_loss = {}
        tot_loss = 0
        outputs = {}

        for task_id, target in enumerate(self.target):
            if target == 'scalar':
                output = scalar
                y = data.y
            elif target == 'vector':
                output = vector
                if not self.dynamics:
                    y = data.dy  # Force prediction, dy is the label
                else:
                    y = data.v_label  # Dynamics prediction, v_label is the label
            else:
                output, y = None, None
            if len(output.shape) == 2:
                output = output.squeeze(-1)
            if len(y.shape) == 2:
                y = y.squeeze(-1)
            if self.return_outputs:
                outputs[target] = (output, y)
            loss = self.loss(output, y)
            if len(loss.shape) == 2:
                loss = loss.mean(dim=-1)
            if self.scalar_pooling and target == 'vector':  # Specifically for energy force prediction
                loss = global_mean_pool(loss, data.batch)
            tot_loss = tot_loss + self.loss_weight[task_id] * loss
            all_loss['loss_' + target] = loss

        return tot_loss, all_loss, outputs

