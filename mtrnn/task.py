import torch

from .network import MTRNN


class Task:
    """
    :param model:  a Network instance.
    :param w:      the relative weight of the two loss terms (the metaprior),
                   :math:`L = w L_x + (1 - w) L_z`.
    :type  w:      float in [0, 1]
    :param learning_rate:  the learning rate of the optimizer.
    :param loss_fun:  either 'KLDiv' or 'MSE', for using KL divergence or
                      mean-squared error as the loss function for the
                      recontruction error.
    :param clip:   if not None, gradient will be normalized at this value
                   before the optimization step.
    :param logs:   Logs instance.
    """
    _EPS = 1e-7

    def __init__(self, model: MTRNN, w: float=0.001,
                       grad_clip:float=0.25, grad_clip_norm_type=2,
                       learning_rate: float=0.001, loss_fun='MSE', logs=None):
        self.model         = model
        self.w             = w
        self.grad_clip     = grad_clip
        self.grad_clip_norm_type = grad_clip_norm_type
        self.learning_rate = learning_rate
        self.logs = logs

        assert 0 <= w <= 1

        # loss function
        if loss_fun == 'MSE':
            self._loss_fun = torch.nn.MSELoss(size_average=False, reduce=True)
        elif loss_fun == 'KLDiv':
            self._loss_fun = torch.nn.KLDivLoss(size_average=False, reduce=True)
        else:
            raise ValueError(("{} is not supported as a loss function; it must"
                              " be either 'MSE' or 'KLDiv'").format(loss_fun))
        self.loss_fun = loss_fun

        # optimizer
        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=self.learning_rate)

    def train(self, batch):
        """Train the network on a batch.

        Backpropagate one time, after all samples have been processed.

        :param batch:  3-tuple: `(indexes, inputs, targets)`, with `indexes` the
                       list of indexes, `inputs` and `target` are tensors of
                       shape :math:`(T, M, N_in)`.
        """
        self.model.train()
        self.optimizer.zero_grad()
        self.model.L_z = 0

        indexes, inputs, target = batch
        self.model.forward_init(indexes)

        outputs = self.model(inputs)
        if self.logs is not None:
            self.logs.log_array('open_loop', (indexes, outputs))

        L_x = 0
        if self.loss_fun == 'KLDiv':
            L_x += self._loss_fun(torch.log(outputs + self._EPS), target)
        else:
            L_x += self._loss_fun(outputs, target)
        L_z =  - self.model.L_z
        L = self.w * L_z + (1 - self.w) * L_x

        L.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.grad_clip,
                                           norm_type=self.grad_clip_norm_type)
        self.optimizer.step()

        n = len(batch)
        return inputs, target, outputs, L/n, L_z/n, L_x/n

    def open_loop(self, batch):
        """Generate closed_loop trajectories

        :param batch:  3-tuple: `(indexes, inputs, targets)`, with `indexes` the
                       list of indexes, `inputs` and `target` are tensors of
                       shape :math:`(T, M, N_in)`.
        """
        indexes, inputs, target = batch
        self.optimizer.zero_grad()
        self.model.L_z = 0

        self.model.eval()
        self.model.forward_init(indexes)
        return self.model.forward(inputs)

    def closed_loop(self, batch):
        """Generate closed_loop trajectories

        :param batch:  3-tuple: `(indexes, inputs, targets)`, with `indexes` the
                       list of indexes, `inputs` and `target` are tensors of
                       shape :math:`(T, M, N_in)`.
        """
        indexes, inputs, target = batch
        I_0 = inputs[0]

        self.optimizer.zero_grad()
        self.model.L_z = 0

        self.model.eval()
        self.model.forward_init(indexes)
        return self.model.forward_closed_loop(I_0, len(inputs))

    def test_closed_loop(self, batch, size_average=False):
        """Test the network for closed-loop generation.

        :param batch:  3-tuple: `(indexes, inputs, targets)`, with `indexes` the
                       list of indexes, `inputs` and `target` are tensors of
                       shape :math:`(T, M, N_in)`.

        Note that since we operate in closed loop, the output are the next
        timestep's input, and therefore the targets should be equal to the
        inputs. The target value will be ignored, and inputs will be used in its
        stead.
        """
        indexes, inputs, target = batch
        outputs = self.closed_loop(batch)
        if self.logs is not None:
            self.logs.log_array('closed_loop', (indexes, outputs))
        return torch.nn.functional.mse_loss(outputs, target,
                    size_average=size_average, reduce=True).item()
