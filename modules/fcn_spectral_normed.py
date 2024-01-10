#code from https://github.com/bunkerj/normalizing-flows
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

import torch
import torch.nn.functional as F


class SpectralNormFC:
    def __init__(self, eps=1e-12):
        self.u = None
        self.eps = eps

    def get_spectral_norm(self, weight, n_spectral_iter, dtype=torch.float32):
        with torch.no_grad():
            if (self.u is None) or (weight.T.shape[1] != self.u.shape[0]):
                self.u = F.normalize(torch.rand(weight.T.shape[1], 1, dtype=dtype, device=weight.device), dim=0, eps=self.eps)
            for _ in range(n_spectral_iter):
                v = F.normalize(weight.T.conj() @ self.u, dim=0, eps=self.eps)
                self.u = F.normalize(weight @ v, dim=0, eps=self.eps)
            spectral_norm = self.u.T.conj() @ weight @ v

        if dtype == torch.cfloat:
            spectral_norm = spectral_norm.norm()

        return spectral_norm

    def get_spectral_norm_multi(self, weights, n_spectral_iter, dtype=torch.float32):
        # TODO: This could possibly be used instead of the above function (perform speed test)
        n_matrices, _, n_cols = weights.shape
        with torch.no_grad():
            if (self.u is None) or (weights.shape[1] != self.u.shape[0]):
                self.u = F.normalize(torch.rand(n_matrices, n_cols, dtype=dtype, device=weights.device), dim=1, eps=self.eps)
            for _ in range(n_spectral_iter):
                v = F.normalize(torch.einsum('ijk,ik->ij', weights.conj(), self.u), dim=1, eps=self.eps)
                self.u = F.normalize(torch.einsum('ijk,ij->ik', weights, v), dim=1, eps=self.eps)
            spectral_norm = torch.einsum('ik,ijk,ij->i', self.u.conj(), weights, v)

        if dtype == torch.cfloat:
            spectral_norm = spectral_norm.abs()

        return spectral_norm.view(-1, 1, 1)

class LinearConstrained(nn.Module):
    def __init__(self, input_dim, output_dim, coeff, n_spectral_iter, bias=True, dtype=torch.float32):
        super().__init__()
        self.coeff = coeff
        self.spectral_norm_fc = SpectralNormFC()
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim, dtype=dtype))
        self.b = nn.Parameter(torch.empty(output_dim, dtype=dtype)) if bias else None
        self.dtype = dtype
        self.n_spectral_iter = n_spectral_iter
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.b is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        W = self._get_W()
        return F.linear(x, W, self.b)

    def _get_W(self):
        n_spectral_iter = self.n_spectral_iter
        spectral_norm = self.spectral_norm_fc.get_spectral_norm(self.weight, n_spectral_iter, dtype=self.dtype)
        scale_term = torch.min(torch.ones(1, device=self.weight.device), self.coeff / spectral_norm)
        return scale_term * self.weight


class CReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return apply_c_relu(x)


def apply_c_relu(x):
    return torch.complex(torch.nn.functional.relu(x.real), torch.nn.functional.relu(x.imag))

class FCNSpectralNormed(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_out, n_hidden_layers, 
                 coeff, n_spectral_iter, dtype=torch.float):
        super().__init__()
        self.net = self._init_net(
            dim_input, 
            dim_hidden, 
            dim_out, 
            n_hidden_layers, 
            coeff, 
            n_spectral_iter, 
            dtype,
        )
        
    def forward(self, x):
        y = self.net(x)
        return y

    def _init_net(self, dim_input, dim_hidden, dim_out, n_hidden_layers, coeff, n_spectral_iter, dtype):
        act = CReLU if dtype == torch.cfloat else nn.ReLU
        layers = []
        for _ in range(n_hidden_layers):
            layers.append(LinearConstrained(dim_input, dim_hidden, coeff, n_spectral_iter, dtype=dtype))
            layers.append(act())
            dim_input = dim_hidden
        layers.append(LinearConstrained(dim_input, dim_out, coeff, n_spectral_iter, dtype=dtype))
        return nn.Sequential(*layers)