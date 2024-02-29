import torch.nn as nn
import torch
from modules.fno import FNO2d

class CouplingLayer(nn.Module):
    def __init__(self, partition, free_net, c_in, device=torch.device('cuda')):
        super().__init__()
        self.free_net = free_net
        self.scaling_factor = nn.Parameter(torch.zeros(c_in))
        self.epsilon = 1e-5
        self.partition = partition
        self.device = device
    def forward(self, x, sample=False, condition=None):
        mask = self.partition.to(self.device)
        x_left = x * mask
        if condition is not None:
            st = self.free_net(torch.cat([x_left, condition], dim=1))
        else:
            st = self.free_net(x_left)
        s, t = st.chunk(2, dim=1)
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        s = torch.tanh(s / s_fac) * s_fac
        s = s * (1 - mask)
        t = t * (1 - mask)
        if not sample:
            z = x * torch.exp(s) + t
            ldj = s.sum(dim=[1, 2, 3])
        else:
            z = (x - t) * torch.exp(-s)
            ldj = - s.sum(dim=[1, 2, 3])
        return z, ldj

class CouplingLayerScaleInv(nn.Module):
    def __init__(self, partition_even, device, free_net, c_in, use_unet=False):
        super().__init__()
        self.free_net = free_net
        self.use_unet=use_unet
        self.scaling_factor = nn.Parameter(torch.zeros(c_in))
        self.epsilon = 1e-5
        self.partition = partition_even[0]
        self.even = partition_even[1]
        self.device = device
    def forward(self, x, sample=False, condition=None):
        shape = x.shape
        mask = self.partition(shape, self.even).to(self.device)
        x_left = x * mask
        if condition is not None:
            st = self.free_net(torch.cat([x_left, condition], dim=1), shape=shape)
            s, t = st.chunk(2, dim=1)
        else:
            st = self.free_net(x_left, shape=shape)
            s, t = st.chunk(2, dim=1)
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        s = torch.tanh(s / s_fac) * s_fac
        s = s * (1 - mask)
        t = t * (1 - mask)
        if not sample:
            z = x * torch.exp(s) + t
            ldj = s.sum(dim=[1, 2, 3])
        if sample:
            z = (x - t) * torch.exp(-s)
            ldj = None
        return z, ldj     

class CouplingLayer1D(nn.Module):
    def __init__(self, partition, free_net, device=torch.device('cuda')):
        super().__init__()
        self.free_net = free_net
        self.scaling_factor = nn.Parameter(torch.zeros(1))
        self.epsilon = 1e-5
        self.partition = partition
        self.device = device
    def forward(self, x, sample=False, condition=None):
        mask = self.partition.to(self.device)
        x_left = x * mask
        if condition is not None:
            st = self.free_net(torch.cat([x_left, condition], dim=1))
        else:
            st = self.free_net(x_left)
        s, t = st.chunk(2, dim=-1)
        s_fac = self.scaling_factor.exp().view(1, 1)
        s = torch.tanh(s / s_fac) * s_fac
        s = s * (1 - mask)
        t = t * (1 - mask)
        if not sample:
            z = x * torch.exp(s) + t
            ldj = s.sum(dim=-1)
        else:
            z = (x - t) * torch.exp(-s)
            ldj = - s.sum(dim=-1)
        return z, ldj 
    
class CouplingLayerFNO(nn.Module):
    def __init__(self, even, c_in, modes, net_cin, net_cout, net_width):
        super().__init__()
        self.scaling_factor = nn.Parameter(torch.zeros(c_in))
        self.epsilon = 1e-6
        self.even = even
        self.net = FNO2d(modes, modes//2, net_width, net_cin, net_cout)
    def forward(self, x, r, sample=False, condition=None):
        x_left, x_right = self.partition(x, space=True)
        x_unc = x_left if self.even else x_right
        x_ch = x_right if self.even else x_left
        if condition is not None:
            st = self.net(torch.cat([x_unc, condition], dim=1), r)
        else:
            st = self.net(x_unc, r)
        st_left, st_right = self.partition(st, space=True)
        st = st_right if self.even else st_left
        st = torch.fft.rfft2(st, dim=(-2, -1))
        s, t = st.chunk(2, dim=1)
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        s = torch.tanh(s / s_fac) * s_fac
        x_ch = torch.fft.rfft2(x_ch, dim=(-2, -1))
        if not sample:
            z = x_ch * torch.exp(s) + t
            ldj = torch.abs(s).sum(dim=[1, 2, 3])*2
            z = torch.fft.irfft2(z, s=(x.size(-2), x.size(-1)))
        else:
            z = (x_ch - t) * torch.exp(-s)
            ldj = - torch.abs(s).sum(dim=[1, 2, 3])*2
            z = torch.fft.irfft2(z, s=(x.size(-2), x.size(-1)))
        return z + x_unc, ldj
    def partition(self, x, space=False):
        x_f = torch.fft.rfft2(x, dim=(-2, -1))
        size = x.shape[-2]
        x_left = x_f.clone()
        x_right = x_f.clone()
        x_left[..., size//2:, :] = 0
        x_right[..., :size//2, :] = 0
        if space:
            x_left = torch.fft.irfft2(x_left, s=(x.size(-2), x.size(-1)))
            x_right = torch.fft.irfft2(x_right, s=(x.size(-2), x.size(-1)))
        return x_left, x_right