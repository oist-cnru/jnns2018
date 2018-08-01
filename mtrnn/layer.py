import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from . import utils


class MTLayer(nn.Module):
    """
    Multi-timescales layer.

    The forward pass is computed as:

    .. math::
        \\begin{align}
        \\pmb{z}^k_{t} =& \\pmb{\\mu}^k_{t} + \\pmb{\\sigma}^k_{t}\\pmb{\\varepsilon}^k_{t}\\\\
        \\pmb{\\mu}^k_{t}    =& \\left(1 - \\frac{1}{\\tau_k}\\right)\\pmb{\\mu}^k_{t-1}
                                    + \\frac{1}{\\tau_k}\\left(  \\pmb{W}^{\\mu d, k-1}\\pmb{d}^{k-1}_{t-1}
                                                            + \\pmb{W}^{\\mu d, k  }\\pmb{d}^{k  }_{t-1}
                                                            + \\pmb{W}^{\\mu d, k+1}\\pmb{d}^{k+1}_{t-1}
                                                      \\right)\\\\
        \\pmb{\\sigma}^k_{t} =& \\left(1 - \\frac{1}{\\tau_k}\\right)\\pmb{\\sigma}^k_{t-1}
                                    + \\frac{1}{\\tau_k}\\left(  \\pmb{W}^{\\sigma d, k-1}\\pmb{d}^{k-1}_{t-1}
                                                            + \\pmb{W}^{\\sigma d, k  }\\pmb{d}^{k  }_{t-1}
                                                            + \\pmb{W}^{\\sigma d, k+1}\\pmb{d}^{k+1}_{t-1}
                                                      \\right)\\\\
        \\end{align}

    For the special case of the initial layer, these equations hold with
    :math:`\\textrm{with }~\\pmb{c}^{0}_{t} = \\pmb{I}^{0}_{t}`.

    :param τ:       the layer time constant.
    :param N:       the number of units in the layer
    :param N_low:   the number of units in the layer bellow
    :param N_high:  the number of units in the layer above
    :param device:  the device to compute on, either 'cuda' or 'cpu'.
    """


    def __init__(self, τ, N, N_low, N_high, indexes=None, device=None, **kwargs):
        super().__init__() # needed for backward prop.
        self.device = utils.autodevice(device)

        # keeping a copy of the parameters
        self.τ, self.N, self.N_low, self.N_high = τ, N, N_low, N_high

        self.tanh = torch.nn.Tanh()
        self._init_weights()
        self._create_initial_states(indexes)

    def _init_weights(self):
        """Initialize weight matrices"""
        N_low, N, N_high = self.N_low, self.N, self.N_high
        self.W_μ_low  = nn.Linear(N_low,  N, bias=False)
        self.W_μ_self = nn.Linear(N,      N, bias=True)
        self.W_μ_high = nn.Linear(N_high, N, bias=False) if N_high is not None else None

    def _create_initial_states(self, indexes):
        for index in indexes:
            zeros = torch.zeros(self.N, requires_grad=True, device=self.device)
            setattr(self, 'μ_0_{}'.format(index), Parameter(zeros))

    def forward_init(self, indexes, *args, **kwargs):
        """Set the initial state.

        This should be run before any training occurs.

        :param indexes:  list of index of the initial state to setup.
        """
        μ_0 = []

        for index in indexes:
            μ_0.append(getattr(self, 'μ_0_' + str(index)))
        self.M = len(μ_0)  # number of samples
        self.μ = torch.stack(μ_0)
        self._forward_activate()

    def _forward_activate(self):
        """Compute z and d"""
        self.z = self.μ
        self.d = self.tanh(self.z)
        self.d_past = self.d

    def forward_step(self, d_low, d_high):
        """Forward pass

        :param d_low:   activation vector of the lower layer.
        :param d_high:  activation vector of the higher layer. Set to `None` for
                        the forward pass of the highest layer.
        """
        self.μ = ((1 - 1/self.τ) * self.μ
                  + (1/self.τ)   * (   self.W_μ_low(d_low)
                                    +  self.W_μ_self(self.d_past)
                                    + (self.W_μ_high(d_high) if d_high is not None else 0)))
        self._forward_activate()


class MTLayerLearnTau(MTLayer):
    """
    τ is a single-scalar learnable parameter.

    Here, the learnable scalar is 100th of the actual timescale, to increase the
    learning rate (easier than specifying in the optimizer)
    """

    def __init__(self, τ, N, N_low, N_high, indexes=None, boost=100.0, device=None, clamp=True, **kwargs):
        nn.Module.__init__(self) # needed for backward prop.
        self.device = utils.autodevice(device)

        # keeping a copy of the parameters
        self.N, self.N_low, self.N_high = N, N_low, N_high
        self.boost = boost
        self.clamp = clamp

        self.tanh = torch.nn.Tanh()
        self._init_weights()
        self._create_initial_states(indexes)

        self.τ_aux = Parameter(torch.tensor(float(τ) / self.boost, requires_grad=True,
                                        device=self.device))

    @property
    def τ(self):
        if self.clamp:
            return torch.clamp(self.boost * self.τ_aux, 1.0)
        else:
            return self.boost * self.τ_aux
