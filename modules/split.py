import torch.nn as nn
import torch
class SplitFlow(nn.Module):
    def __init__(self, prior, device, record_z=False):
        super().__init__()
        self.prior = prior
        self.device = device
        self.rz = record_z
        self.last_split = None
    def forward(self, z, sample=False, base=None):
        if not sample:
            z, z_split = z.chunk(2, dim=1)
            ldj = self.prior.log_prob(z_split).sum(dim=[1,2,3])
            ldj = ldj.to(self.device)
            if self.rz:
                self.last_split = z_split
        else:
            if not base:
                z_split = self.prior.sample(sample_shape=z.shape).to(self.device)
            else:
                z_split = base
            if self.rz:
                self.last_split = z_split
            z = torch.cat([z, z_split], dim=1)
            ldj = -self.prior.log_prob(z_split).sum(dim=[1,2,3])
        return z, ldj
    
class SplitFlow1D(nn.Module):
    def __init__(self, prior, device, record_z=False):
        super().__init__()
        self.prior = prior
        self.device = device
        self.rz = record_z
        self.last_split = None
    def forward(self, z, sample=False, base=None):
        if not sample:
            z, z_split = z.chunk(2, dim=1)
            print(z.shape, z_split.shape)
            ldj = self.prior.log_prob(z_split).sum(dim=-1)
            ldj = ldj.to(self.device)
            if self.rz:
                self.last_split = z_split
        else:
            shape = list(z.shape)
            shape[-1] = (shape[-1]//2)*2
            shape = torch.Size(shape)
            if not base:
                z_split = self.prior.sample(sample_shape=shape).to(self.device)
            else:
                z_split = base
            if self.rz:
                self.last_split = z_split
            z = torch.cat([z, z_split], dim=1)
            ldj = -self.prior.log_prob(z_split).sum(dim=-1)
        return z, ldj