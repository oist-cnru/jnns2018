import torch
import torch.nn as nn

from . import utils


class MTRNN(nn.Module):
    """
    Network class.

    :param netparams:     layers configuration.
    :type  netparams:     dicts
    :param datashape_in:  shape of the input data :math:`(T, M, N_in)`.
    :param datashape_out: shape of the input data :math:`(T, M, N_out)`. If None,
                          will be assumed to be equal to `datashape_in`.
    :param log_eps:       epsilon value to avoid infinity explosion of logs.
    :param device:        the device to compute on, either 'cuda' or 'cpu'.

    For each layer, two parameters are required:

    1. `τ`, the time constant.
    2. `N`, the number of units.

    For instance, layer 1 in the paper is paramaterized as:

    .. code-block:: python

        {'τ': 1, 'N': 121}

    The number of unit in the first layer is assumed to be the size of the
    input.
    """
    default_layer = 'mtrnn.MTLayer'

    def __init__(self, netparams, datashape_in, datashape_out=None,
                       batch_size=None, device=None, **kwargs):
        super().__init__() # needed for backward prop.
        self.device = utils.autodevice(device)
        
        if datashape_out is None:
            datashape_out = datashape_in
        self.n_samples, self.T, self.N_in = datashape_in
        self.N_out = datashape_out[-1]
        self.batch_size = batch_size
        self.log_eps = torch.tensor(float(netparams.get('log_eps', 1e-8)))

        self.t = 0

        self.layers = []
        for k, layer_params in enumerate(netparams['layers']):
            N_low  = netparams['layers'][k-1]['N'] if k > 0 else self.N_in
            N_high = netparams['layers'][k+1]['N'] if (k+1) < len(netparams['layers']) else None
            layer_cls = utils.import_str(layer_params.get('cls', self.default_layer))

            layer = layer_cls(N_low=N_low, N_high=N_high,
                              T=self.T, indexes=range(self.n_samples),
                              batch_size=self.batch_size, device=self.device,
                              **layer_params)

            self.layers.append(layer)
            self.add_module('layer_{}'.format(k), layer)

        self._init_weights()
        self.L_z = 0          # regularization loss
        self.to(self.device)  # setting the computing device

    def _init_weights(self):
        """Create the output matrix"""
        self.W_out = nn.Linear(self.layers[0].N, self.N_out, bias=True)

    def _compute_reg_loss_step(self):
        """Compute the regularization loss after one step"""
        L_z_t = 0
        for layer in self.layers:
            L_z_t += - layer.μ.pow(2).sum()
        self.L_z += L_z_t / sum(layer.N for layer in self.layers)

    def _compute_output(self):
        return self.W_out(self.layers[0].d)

    def _forward_step(self, I_t: torch.Tensor):
        """
        Forward pass for one timestep.

        :param I_t: input vector for one step of shape :math:`(M, N_{in})`, with
                    :math:`M` the number of samples in the mini-batch.
        """
        for k, layer in enumerate(self.layers):
            d_low  = self.layers[k-1].d_past if k > 0 else I_t
            d_high = self.layers[k+1].d_past if k+1 < len(self.layers) else None
            layer.forward_step(d_low, d_high)

        self._compute_reg_loss_step()
        return self._compute_output()

    def forward_init(self, *args, **kwargs):
        self.t = 0
        for layer in self.layers:
            layer.forward_init(*args, **kwargs)

    def forward(self, I: torch.Tensor):
        """
        Forward pass for one sample.

        :param I: input vector for one step of shape :math:`(T, M, N_{in})`,
                  with :math:`T` the number of timesteps and :math:`M` the
                  number of samples in the mini-batch.
        """
        output_ts = []
        for t, I_t in enumerate(I):
            self.t = t
            output_ts.append(self._forward_step(I_t.unsqueeze(0)))
            if hasattr(self, 'logs'):
                self.logs.log_network_state(self)
                self.logs.log_τs(self)
        return torch.cat(output_ts)

    def forward_closed_loop(self, I_0, T=None):
        """
        Forward pass in closed loop.

        The output of a timestep is used as the input of the next timestep. For
        this to be possible, :math:`N_{in}` must be equal to :math:`N_{out}`.

        :param I_0: input vector for one timestep, of size :math:`(M, N_{in})`.
        :param T:   number of timesteps for which to generate an output,
                    including `I_0`.
        :return:    an output vector of size `T`.
        """
        T = self.T if T is None else T
        I_t, output_ts = I_0.unsqueeze(0), []
        output_ts.append(I_t)

        for t in range(T-1):
            I_t = self._forward_step(I_t)
            output_ts.append(I_t)
        return torch.cat(output_ts)
