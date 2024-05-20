import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.unet import UNet
from modules.coupling import CouplingLayerScaleInv
from modules.act_norm import ActNorm
from modules.free_net import CNN
from modules.partitions import checkerBoard
class StochFlow(nn.Module):
    def __init__(self, num_coupling_layers, num_steps, device='cuda'):
        super().__init__()
        self.unet = UNet(in_d=1, out_d=2, condition_size=1)
        self.flow = nn.ModuleList([j for i in range(num_coupling_layers) for j in [CouplingLayerScaleInv(partition_even=[checkerBoard, i%2==0], device=device, free_net=CNN(1, c_out=2, kernel_size=3, padding=1), c_in=1)]])
        self.prior = torch.distributions.Normal(0, 1)
        self.scale = nn.Parameter(torch.zeros(1))
        self.num_steps = num_steps
        self.actnorm = nn.Parameter(torch.zeros(num_steps*2))
    def forward(self, x):
        # checkerboard_mask = torch.zeros_like(x)
        # checkerboard_mask[:, 0::2, 0::2] = 1
        # checkerboard_mask[:, 1::2, 1::2] = 1
        # sldj = 0
        # for i in reversed(range(self.num_steps)):
        #     shift_scale = self.actnorm[i*2:i*2+2]
        #     x = x * shift_scale[1].exp() + shift_scale[0]
        #     mask = checkerboard_mask if i % 2 == 0 else 1 - checkerboard_mask
        #     condition = torch.ones_like(x) * i / self.num_steps
        #     condition = torch.cat([condition, mask], dim=1)
        #     condition = condition[..., 0, 0]
        #     ms = self.unet(x * mask, condition)
        #     mu, sig = ms.chunk(2, dim=1)
        #     sig = sig.exp()
        #     base_shape = x.expand(x.shape[0], 2, x.shape[2], x.shape[3]).shape
        #     base = self.prior.sample(base_shape).to(x.device)
        #     ldj = self.prior.log_prob(base).sum(dim=(1, 2, 3))
        #     base = base * sig + mu
        #     ldj += - sig.log().sum(dim=(1, 2, 3))
        #     for f in self.flow:
        #         base, ll = f(base, sample=True)
        #         ldj += ll
        #     s, t = base.chunk(2, dim=1)
        #     s = s * (1 - mask)
        #     t = t * (1 - mask)
        #     s_fac = self.scale.exp().view(1, -1, 1, 1)
        #     s = torch.tanh(s / s_fac) * s_fac
        #     s = s.exp()
        #     x = (x-t) / s
        #     sldj += ldj
        # sldj += self.prior.log_prob(x).sum(dim=(1, 2, 3))
        z = torch.zeros_like(x).to(x.device)
        optimal_diff = x - z
        optimal_shift = optimal_diff / self.num_steps
        zs = [z + i * optimal_shift for i in range(self.num_steps)]
        sldj = 0
        intermed = []
        for i in range(self.num_steps):
            ldj = 0
            z = zs[i]
            sample = optimal_shift
            condition = torch.ones_like(x) * i / self.num_steps
            condition = condition[..., 0, 0]
            ms = self.unet(z, condition)
            mu, sig = ms.chunk(2, dim=1)
            sig = sig.exp()
            for f in self.flow:
                sample, ll = f(sample)
                ldj += ll
            sample = (sample - mu) / sig
            ldj += self.prior.log_prob(sample).sum(dim=(1, 2, 3))
            sldj += ldj
        return -sldj.sum()
    def sample(self, num_samples, resolution, device):
        x = torch.zeros((num_samples, 1, resolution, resolution)).to(device)
        # checkerboard_mask = torch.zeros_like(x)
        # checkerboard_mask[:, 0::2, 0::2] = 1
        # checkerboard_mask[:, 1::2, 1::2] = 1
        # z = torch.randn_like(x).to(x.device)
        # for i in range(self.num_steps):
        #     shift_scale = self.actnorm[i*2:i*2+2]
        #     z = (z - shift_scale[0]) / shift_scale[1].exp()
        #     mask = checkerboard_mask if i % 2 == 0 else 1 - checkerboard_mask
        #     condition = torch.ones_like(x) * i / self.num_steps
        #     condition = torch.cat([condition, mask], dim=1)
        #     condition = condition[..., 0, 0]
        #     ms = self.unet(z * mask, condition)
        #     mu, sig = ms.chunk(2, dim=1)
        #     sig = sig.exp()
        #     base_shape = x.expand(x.shape[0], 2, x.shape[2], x.shape[3]).shape
        #     base = self.prior.sample(base_shape).to(x.device)
        #     base = base * sig + mu
        #     for f in self.flow:
        #         base, _ = f(base, sample=True)
        #     s, t = base.chunk(2, dim=1)
        #     s = s * (1 - mask)
        #     t = t * (1 - mask)
        #     s_fac = self.scale.exp().view(1, -1, 1, 1)
        #     s = torch.tanh(s / s_fac) * s_fac
        #     s = s.exp()
        #     z = z * s + t
        # z = F.sigmoid(z) * 255
        # return z
        for i in range(self.num_steps):
            condition = torch.ones_like(x) * i / self.num_steps
            condition = condition[..., 0, 0]
            ms = self.unet(x, condition)
            mu, sig = ms.chunk(2, dim=1)
            sig = sig.exp()
            sample = self.prior.sample(x.shape).to(x.device)
            sample = sample * sig + mu
            for f in reversed(self.flow):
                sample, _ = f(sample, sample=True)
            x += sample
        return x
    



            
            



