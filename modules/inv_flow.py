#code from https://github.com/bunkerj/normalizing-flows
import torch
import torch.nn as nn
from utils.ldj import get_ldj
from modules.fcn_spectral_normed import FCNSpectralNormed

class InvertibleResNetLayerBase(nn.Module):
    def __init__(self, n_ldj_iter, n_exact_terms, n_fixed_iter, use_unbiased):
        super().__init__()
        self.n_ldj_iter = n_ldj_iter
        self.n_exact_terms = n_exact_terms
        self.n_fixed_iter = n_fixed_iter
        self.use_unbiased = use_unbiased
        self.g = None # To be implemented in child class

    def forward(self, x, sample=False):
        if not sample:
            with torch.enable_grad():
                x = x.requires_grad_(True)
                g_x, ldj = self._get_output_and_ldj(x)
                z = x + g_x
        else:
            z = self._get_inverse(x)
            ldj = None
        return z, ldj

    def _get_output_and_ldj(self, x):
        n_ldj_iter = self.n_ldj_iter
        n_exact_terms = self.n_exact_terms

        if not self.use_unbiased:
            g_x, ldj = get_ldj(
                nnet=self.g, 
                x=x, 
                n_ldj_iter=n_ldj_iter, 
                n_exact_terms=n_exact_terms, 
                est_name='neumann',
                mem_eff=True, 
                is_training=self.training)
        else:
            raise NotImplementedError

        return g_x, ldj

    def _get_inverse(self, z):
        n_fixed_point_iter = self.n_fixed_iter
        with torch.no_grad():
            x = z
            for _ in range(n_fixed_point_iter):
                x = z - self.g(x)
        return x 

class SpectralResnetLayer(InvertibleResNetLayerBase):
    def __init__(self, dim_hidden, depth, k_max, n_fourier_coeffs, coeff, n_ldj_iter, n_exact_terms, n_fixed_iter, use_unbiased, n_spectral_iter, **kwargs):
        super().__init__(n_ldj_iter, n_exact_terms, n_fixed_iter, use_unbiased)
        self.g = SpectralResnetLayerForward(dim_hidden, depth, k_max, n_fourier_coeffs, coeff, n_spectral_iter)


class SpectralResnetLayerForward(nn.Module):
    def __init__(self, dim_hidden, depth, k_max, n_fourier_coeffs, coeff, n_spectral_iter):
        super().__init__()
        dim = self._get_dim(k_max, n_fourier_coeffs)
        self.k_max = k_max
        self.n_fourier_coeffs = n_fourier_coeffs
        self.fcn = FCNSpectralNormed(
            dim_input=dim, 
            dim_hidden=dim_hidden, 
            dim_out=dim, 
            n_hidden_layers=depth, 
            coeff=coeff, 
            n_spectral_iter=n_spectral_iter, 
            dtype=torch.cfloat,
        )

    def forward(self, x):
        F_in = torch.fft.rfft2(x, norm='ortho')
        F_out = self._transform_modes(F_in)
        x_out = torch.fft.irfft2(F_out, norm='ortho', s=x.shape[-2:])
        return x_out

    def _get_dim(self, k_max, n_fourier_coeffs):
        if n_fourier_coeffs is None:
            return (k_max + 1) ** 2 + k_max * (k_max + 1) 
        else:
            return n_fourier_coeffs

    def _transform_modes(self, F_in):
        if self.n_fourier_coeffs is None:
            return self._transform_low_modes(F_in)
        else:
            return self._transform_all_modes(F_in)

    def _transform_low_modes(self, F_in):
        block_in_a = F_in[:, :, :self.k_max+1, :self.k_max+1].flatten(-3)
        block_in_b = F_in[:, :, -self.k_max:, :self.k_max+1].flatten(-3)
        modes_in = torch.cat([block_in_a, block_in_b], dim=-1)

        F_low_modes_new = self.fcn(modes_in)

        b, c = F_in.shape[:2]
        F_out = torch.zeros_like(F_in)
        block_out_a = F_low_modes_new[:, :(self.k_max+1)**2].view(b, c, self.k_max+1, self.k_max+1)
        block_out_b = F_low_modes_new[:, (self.k_max+1)**2:].view(b, c, self.k_max, self.k_max+1)
        
        F_out[:, :, :self.k_max+1, :self.k_max+1] = block_out_a
        F_out[:, :, -self.k_max:, :self.k_max+1] = block_out_b
        return F_out

    def _transform_all_modes(self, F_in):
        F_in_flat = F_in.flatten(-2)
        F_out_flat = self.fcn(F_in_flat)
        F_out = F_out_flat.view(F_in.shape)
        return F_out