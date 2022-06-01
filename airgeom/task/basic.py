from torch.nn import Module


class BasicTask(Module):
    """
    The basic task.

    :param rep: the representation module.
    """
    def __init__(self, rep):
        super(BasicTask, self).__init__()
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

    def forward(self, data):
        """
        Forward passing with the data object.

        :param data: The data object.
        :return: The loss computed.
        """
        raise NotImplementedError()