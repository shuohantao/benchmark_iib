from typing import Any, Mapping
import torch 
import torch.nn as nn


class ActNorm(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.register_buffer("is_initialized", torch.tensor([False]))

    def forward(self, z, sample=False, **kwargs):
        move_towards_base = not sample
        if move_towards_base:
            if not self.is_initialized.item():
                self._initialize(z)
            z = torch.exp(self.log_scale) * z + self.shift
        else:
            if not self.is_initialized.item():
                self._initialize(z)
            z = (z - self.shift) / torch.exp(self.log_scale)

        ldj = self._get_log_determinant_jacobian(z) if move_towards_base else -self._get_log_determinant_jacobian(z)
        return z, ldj

    def _get_log_determinant_jacobian(self, z):
        h, w = z.shape[-2:]
        ldj = (h * w)*self.log_scale.sum() * torch.ones(z.shape[0], device=z.device)
        return ldj
    def _initialize_reverse(self, z):
        with torch.no_grad():
            std = z.std([0, 2, 3]).view(1, z.shape[1], 1, 1)
            mu = (z/std).mean([0, 2, 3]).view(1, z.shape[1], 1, 1)
            self.log_scale.data.copy_(torch.log(std + 1e-6))
            self.shift.data.copy_(mu)
    def _initialize(self, z):
        with torch.no_grad():
            std = z.std([0, 2, 3]).view(1, z.shape[1], 1, 1)
            mu = (z/std).mean([0, 2, 3]).view(1, z.shape[1], 1, 1)
            self.log_scale.data.copy_(-torch.log(std + 1e-6))
            self.shift.data.copy_(-mu)
            self.is_initialized = torch.Tensor([True])
    
    def load_state_dict(self, state_dict, strict=True):
        self.is_initialized = True
        super().load_state_dict(state_dict, strict)

class ActNormAug(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(1, 1, ndim))
        self.shift_aug = nn.Parameter(torch.zeros(1, 1, ndim))
        self.log_scale = nn.Parameter(torch.zeros(1, 1, ndim))
        self.log_scale_aug = nn.Parameter(torch.zeros(1, 1, ndim))
        self.is_initialized = False
        self.ndim = ndim
    def forward(self, x, z, sample=False, **kwargs):
        if not sample:
            if not self.is_initialized:
                self._initialize(x, z)
            z = torch.exp(self.log_scale_aug) * z + self.shift_aug
            x = torch.exp(self.log_scale) * x + self.shift
        else:
            # assert self.is_initialized
            z = (z - self.shift_aug) / torch.exp(self.log_scale_aug)
            x = (x - self.shift) / torch.exp(self.log_scale)
        ldj = self._get_log_determinant_jacobian(x, z).unsqueeze(-1)
        ldj = - ldj if sample else ldj
        return x, z, ldj

    def _get_log_determinant_jacobian(self, x, z):
        ldj = (self.log_scale.sum() + self.log_scale_aug.sum()) * torch.ones(x.shape[:-1], device=x.device)
        return ldj

    def _initialize(self, x, z):
        with torch.no_grad():
            std = x.std([0, 1]).view(1, 1, self.ndim)
            mu = (x/std).mean([0, 1]).view(1, 1, self.ndim)
            self.log_scale.data.copy_(-torch.log(std + 1e-6))
            self.shift.data.copy_(-mu)
            std = z.std([0, 1]).view(1, 1, self.ndim)
            mu = (z/std).mean([0, 1]).view(1, 1, self.ndim)
            self.log_scale_aug.data.copy_(-torch.log(std + 1e-6))
            self.shift_aug.data.copy_(-mu)
            self.is_initialized = True

class ActNormSeq(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(1, 1, n_channels))
        self.log_scale = nn.Parameter(torch.zeros(1, 1, n_channels))
        self.register_buffer("is_initialized", torch.tensor([False]))
        self.n_channels = n_channels
    def forward(self, z, sample=False, **kwargs):
        move_towards_base = not sample
        if move_towards_base:
            if not self.is_initialized.item():
                self._initialize(z)
            z = torch.exp(self.log_scale) * z + self.shift
        else:
            if not self.is_initialized.item():
                self._initialize_reverse(z)
            z = (z - self.shift) / torch.exp(self.log_scale)

        ldj = self._get_log_determinant_jacobian(z) if move_towards_base else -self._get_log_determinant_jacobian(z)
        return z, ldj

    def _get_log_determinant_jacobian(self, z):
        ldj = z.shape[0] * self.log_scale.sum() * torch.ones(z.shape[1], device=z.device)
        return ldj

    def _initialize(self, x):
        with torch.no_grad():
            std = x.std([0, 1]).view(1, 1, self.n_channels)
            mu = (x/std).mean([0, 1]).view(1, 1, self.n_channels)
            self.log_scale.data.copy_(-torch.log(std + 1e-6))
            self.shift.data.copy_(-mu)
            self.is_initialized = torch.Tensor([True])
    def _initialize_reverse(self, x):
        with torch.no_grad():
            std = x.std([0, 1]).view(1, 1, self.n_channels)
            mu = (x/std).mean([0, 1]).view(1, 1, self.n_channels)
            self.log_scale.data.copy_(torch.log(std + 1e-6))
            self.shift.data.copy_(mu)
            self.is_initialized = torch.Tensor([True])

class ActNormParam(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.register_buffer("is_initialized", torch.tensor([False]))
        self.shift.requires_grad_(False)
        self.log_scale.requires_grad_(False)
        self.n_channels = n_channels
    def forward(self, z, param, sample=False, **kwargs):
        con = param
        log_scale_mod, shift_mod = torch.chunk(con, 2, dim=0)
        log_scale_mod = log_scale_mod.view(1, self.n_channels, 1, 1)
        shift_mod = shift_mod.view(1, self.n_channels, 1, 1)
        move_towards_base = not sample
        if move_towards_base:
            if not self.is_initialized.item():
                self._initialize(z)
            z = torch.exp(self.log_scale+log_scale_mod) * z + self.shift + shift_mod
        else:
            if not self.is_initialized.item():
                self._initialize(z)
            z = (z - self.shift - shift_mod) / torch.exp(self.log_scale + log_scale_mod)

        ldj = self._get_log_determinant_jacobian(z, con) if move_towards_base else -self._get_log_determinant_jacobian(z, con)
        return z, ldj

    def _get_log_determinant_jacobian(self, z, con):
        log_scale_mod, shift_mod = torch.chunk(con, 2, dim=0)
        log_scale_mod = log_scale_mod.view(1, self.n_channels, 1, 1)
        shift_mod = shift_mod.view(1, self.n_channels, 1, 1)
        log_scale = self.log_scale + log_scale_mod
        h, w = z.shape[-2:]
        ldj = (h * w)*log_scale.sum() * torch.ones(z.shape[0], device=z.device)
        return ldj
    def _initialize_reverse(self, z):
        with torch.no_grad():
            std = z.std([0, 2, 3]).view(1, z.shape[1], 1, 1)
            mu = (z/std).mean([0, 2, 3]).view(1, z.shape[1], 1, 1)
            self.log_scale.data.copy_(torch.log(std + 1e-6))
            self.shift.data.copy_(mu)
    def _initialize(self, z):
        with torch.no_grad():
            std = z.std([0, 2, 3]).view(1, z.shape[1], 1, 1)
            mu = (z/std).mean([0, 2, 3]).view(1, z.shape[1], 1, 1)
            self.log_scale.data.copy_(-torch.log(std + 1e-6))
            self.shift.data.copy_(-mu)
            self.is_initialized = torch.Tensor([True])
    
    def load_state_dict(self, state_dict, strict=True):
        self.is_initialized = True
        super().load_state_dict(state_dict, strict)