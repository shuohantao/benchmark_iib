import torch.nn as nn
import torch


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