import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys; sys.path.append(os.path.abspath(os.path.join('..'))); sys.path.append(os.path.abspath(os.path.join('.')))
from torch.distributions.normal import Normal
from modules.squeeze import SqueezeFlow
from modules.split import SplitFlow
from modules.coupling import CouplingLayerScaleInv, CouplingLayerFNOv2
from modules.dequantization import Dequantization
from modules.conv import InvConv2d
from modules.free_net import CNN_Linear
from modules.filters import bilinear_seg
from modules.act_norm import ActNorm
from modules.partitions import checkerBoard

class CFFNO(nn.Module):
    def __init__(self, shape, depth=16, width=64, modes = 16, prior=Normal(0, 1), **kwargs):
        super().__init__()
        self.prior = prior
        self.flow = nn.ModuleList()
        self.flow.append(Dequantization())
        self.modes = modes
        even = True
        for i in range(depth):
            self.flow.append(ActNorm(1))
            self.flow.append(CouplingLayerFNOv2(shape=shape, even=even, c_in=1, modes=modes, net_cin=1, net_cout=2, net_width=width))
            even = not even
    def forward(self, x):
        sldj = 0
        for j, i in enumerate(self.flow):
            x, ldj = i(x)
            sldj += ldj
        sldj = sldj.sum() + self.prior.log_prob(x).sum()
        return -sldj
    def sample(self, num_samples, resolution,  device, prior=None, **kwargs):
        with torch.no_grad():
            prior_shape = resolution
            if prior is not None:
                z = prior
            else:
                z = self.prior.sample(torch.Size([num_samples]+[1, prior_shape, prior_shape])).to(device)
            for i in reversed(self.flow):
                z, _ = i(z, sample=True)
            z = z
            return z