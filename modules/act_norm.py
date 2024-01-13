import torch 
import torch.nn as nn


class ActNorm(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.is_initialized = False

    def forward(self, z, move_towards_base=True):
        if move_towards_base:
            if not self.is_initialized:
                self._initialize(z)
            z = torch.exp(self.log_scale) * z + self.shift
        else:
            assert self.is_initialized
            z = (z - self.shift) / torch.exp(self.log_scale)

        ldj = self._get_log_determinant_jacobian(z) if move_towards_base else None
        return z, ldj

    def _get_log_determinant_jacobian(self, z):
        h, w = z.shape[-2:]
        ldj = (h * w)*self.log_scale.sum() * torch.ones(z.shape[0], device=z.device)
        return ldj

    def _initialize(self, z):
        with torch.no_grad():
            std = z.std([0, 2, 3]).view(1, z.shape[1], 1, 1)
            mu = (z/std).mean([0, 2, 3]).view(1, z.shape[1], 1, 1)
            self.log_scale.data.copy_(-torch.log(std + 1e-6))
            self.shift.data.copy_(-mu)
            self.is_initialized = True
