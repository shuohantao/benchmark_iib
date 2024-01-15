#modified from https://github.com/bunkerj/normalizing-flows
import torch
import torch.nn as nn
from modules.dequantization import Dequantization
from modules.inv_flow import SpectralResnetLayer
from modules.act_norm import ActNorm
from torch.distributions.normal import Normal
from utils.base_dist import BaseDistManager

class ResSNO(nn.Module):
    def __init__(self, n_layers, dim_hidden, depth, k_max, n_fourier_coeffs, coeff, n_ldj_iter, n_exact_terms, n_fixed_iter, use_unbiased, n_spectral_iter, type, ls=None, var=None, eps=None, **kwargs):
        super().__init__()
        self.flows = nn.ModuleList()
        self.flows.append(Dequantization())
        for i in range(n_layers):
            self.flows.append(ActNorm(1))
            self.flows.append(SpectralResnetLayer(dim_hidden, depth, k_max, n_fourier_coeffs, coeff, n_ldj_iter, n_exact_terms, n_fixed_iter, use_unbiased, n_spectral_iter, **kwargs))
        self.base_dist_manager = BaseDistManager(type, ls, var, eps)
        self.anti_aliasing_manager = None
    def forward(self, z, base_shape=None):
        if base_shape is None:
            base_shape = z.shape[-3:]
        return -self._get_log_likelihood(z, base_shape)
    
    def get_latent_from_data(self, x):
        return self._pass_through_flow(x, sample=False)[0]

    def get_data_from_latent(self, z):
        return self._pass_through_flow(z, sample=True)[0]

    def get_flow_trajectory(self, z, flow_trajectory_interval, image_shape, sample=False):
        _, _, flow_trajectory = self._pass_through_flow(z, sample, flow_trajectory_interval, image_shape)
        return flow_trajectory

    def sample(self, num_samples, resolution, scale_factor=1, **kwargs):
        base_shape = (1, resolution, resolution)
        base_dist = self.base_dist_manager.get_base_dist(base_shape)
        z = base_dist.sample((num_samples * base_shape[0],))
        z = z.view(num_samples, *base_shape)
        z, _, _ = self._pass_through_flow(z, sample=True)
        z = self._scale(z, z.shape[-2] * scale_factor)
        z = ((z - z.min()) / (z.max() - z.min()) * 255).int()
        return z

    def _pass_through_flow(self, z_in, sample, 
                           flow_trajectory_interval=None, image_shape=None):
        if self.anti_aliasing_manager is not None:
            if not sample:
                n_modes = self.anti_aliasing_manager.n_modes_target
                z_in = self._scale(z_in, n_modes)
                z_in = z_in.int()
            else:
                n_modes = self.anti_aliasing_manager.n_modes_base
                z_in = self._scale(z_in, n_modes)

        z = z_in
        ldj_sum = 0
        flow_trajectory = []
        flows = self.flows if not sample else reversed(self.flows)
        for i, flow in enumerate(flows):
            if isinstance(flow, ActNorm):
                z, ldj = flow(z, move_towards_base=not sample)
            else:
                z, ldj = flow(z, sample=sample)
            if not sample:
                ldj_sum += ldj
            if flow_trajectory_interval is not None and i % flow_trajectory_interval == 0:
                flow_trajectory.append(z.view(1, *image_shape).detach().clone().cpu())

        return z, ldj_sum, flow_trajectory
   
    def _get_log_likelihood(self, z, base_shape):
        z, ldj_sum, _ = self._pass_through_flow(z, sample=False)
        base_dist = self.base_dist_manager.get_base_dist(base_shape)
        log_likelihood = base_dist.log_prob(z.view(z.shape[0] * z.shape[1], -1)).sum() + ldj_sum.sum()
        return log_likelihood

    def _scale(self, z, n_modes):
        if z.shape[-2] > n_modes:
            z_out = self.anti_aliasing_manager.downsample(z, z.shape[-2] // n_modes)
        elif z.shape[-2] < n_modes:
            z_out = self.anti_aliasing_manager.upsample(z, n_modes // z.shape[-2]) 
        else:
            z_out = z
        return z_out
