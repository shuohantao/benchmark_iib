import torch.nn as nn
import torch
class SplitFlow(nn.Module):
    def __init__(self, prior, device, test_inv=False):
        super().__init__()
        self.prior = prior
        self.device = device
        self.test_inv = test_inv
        self.last_split = None
    def forward(self, z, sample=False):
        if not sample:
            z, z_split = z.chunk(2, dim=1)
            ldj = self.prior.log_prob(z_split).sum(dim=[1,2,3])
            ldj = ldj.to(self.device)
            if self.test_inv:
                self.last_split = z_split
        else:
            if self.test_inv:
                z_split = self.last_split
            else:
                z_split = self.prior.sample(sample_shape=z.shape).to(self.device)
            z = torch.cat([z, z_split], dim=1)
            ldj = -self.prior.log_prob(z_split).sum(dim=[1,2,3])
        return z, ldj