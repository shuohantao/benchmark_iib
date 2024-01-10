import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import RandomState

class LinearConstrained(nn.Module):
    def __init__(self, input_dim, output_dim, coeff, iterative_step_manager, rayleigh_init=True, dtype=torch.float32):
        super().__init__()
        self.coeff = coeff
        self.spectral_norm_fc = SpectralNormFCComplex()
        self.iterative_step_manager = iterative_step_manager
        self.weight = torch.empty(output_dim, input_dim, dtype=dtype)
        if rayleigh_init:
            self.weight = nn.Parameter(self.get_complex_inits(self.weight))
        else:
            self.weight = nn.Parameter(self.weight)
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.b = nn.Parameter(torch.empty(output_dim, dtype=dtype))
        self.dtype = dtype
        if self.b is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        W = self._get_W()
        return F.linear(x, W, self.b)

    def _get_W(self):
        n_spectral_iter = self.iterative_step_manager.get_config()['n_spectral_iter']
        spectral_norm = self.spectral_norm_fc.get_spectral_norm(self.weight, n_spectral_iter, dtype=self.dtype)
        scale_term = torch.min(torch.ones(1, device=self.weight.device), self.coeff / spectral_norm)
        return scale_term * self.weight
    
    def get_complex_inits(self, weight, seed=None, criterion='glorot'):
        "Creates real and imaginary Rayleigh weights for initialization."
        # create random number generator
        rand = RandomState(seed if seed is None else torch.initial_seed())
        # get shape of the weights
        weight_size = weight.size()
        # find number of input and output connection
        fan_in,fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        # find sigma to meet chosen variance criteria
        assert criterion in ('he','glorot')
        factor = fan_in if criterion == 'he' else fan_in + fan_out
        sigma = 1. / np.sqrt(factor)
        # draw rayleigh magnitude samples
        magnitude = rand.rayleigh(scale=sigma, size=weight_size)
        # draw uniform angle samples
        phase = rand.uniform(low=-np.pi, high=np.pi, size=magnitude.shape)
        # split magnitudes into real and imaginary components
        real = magnitude * np.cos(phase)
        imag = magnitude * np.sin(phase)
        # turn into float tensors and return
        real,imag = map(torch.from_numpy, [real,imag])
        return torch.complex(real, imag).to(weight.dtype)

class SpectralNormFCComplex:
    '''
    This should be the correct implementation when `weight` can also be complex.
    All the .T operations are replaced with conjugate transposes, and the
    special case where the .norm() is taken is no longer needed.
    '''

    def __init__(self, eps=1e-12):
        self.u = None
        self.eps = eps

    def get_spectral_norm(self, weight, n_spectral_iter, dtype=torch.float32):
        with torch.no_grad():
            if (self.u is None) or (torch.conj(weight).T.shape[1] != self.u.shape[0]):
                self.u = F.normalize(torch.rand(torch.conj(weight).T.shape[1], 1, dtype=dtype, device=weight.device), dim=0, eps=self.eps)
            for _ in range(n_spectral_iter):
                v = F.normalize(torch.conj(weight).T @ self.u, dim=0, eps=self.eps)
                self.u = F.normalize(weight @ v, dim=0, eps=self.eps)
            spectral_norm = torch.conj(self.u).T @ weight @ v

        #if dtype == torch.cfloat:
        #    spectral_norm = spectral_norm.norm()


        # any remaining complex part is numerical error --- just discard it
        spectral_norm = torch.real(spectral_norm)

        # the .sum() is just a hack to reshape a tensor with 1 element to a scalar
        return spectral_norm.sum()