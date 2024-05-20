import torch.nn as nn
import torch
from modules.act_norm import ActNorm
from modules.coupling import CouplingLayerScaleInv
from modules.split import SplitFlow
from modules.squeeze import SqueezeFlow
from modules.conv import InvConv2d
from modules.free_net import CNN_Linear, CNN
from modules.dequantization import Dequantization
from modules.partitions import checkerBoard

class RealNVP(nn.Module):
    def __init__(self, modes, depth_1=8, depth_2=4, depth_3=8):
        super(RealNVP, self).__init__()
        self.flow = nn.ModuleList()
        self.flow.append(Dequantization())
        self.prior = torch.distributions.Normal(0, 1)
        even = True
        for i in range(depth_1):
            self.flow.append(ActNorm(1))
            self.flow.append(CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, even], free_net= CNN(c_in=1, c_out=2, n_hidden=32, kernel_size=3, padding=1), c_in=1))
            even = not even
        # self.flow.append(SqueezeFlow())
        # for i in range(depth_2):
        #     self.flow.append(ActNorm(4))
        #     self.flow.append(InvConv2d(4))
        #     self.flow.append(CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, even], free_net=CNN(c_in=4, c_out=8, n_hidden=64, kernel_size=3, padding=1), c_in=4))
        #     even = not even
        # self.flow.append(SplitFlow(prior=self.prior, device=torch.device('cuda')))
        # self.flow.append(SqueezeFlow())
        # for i in range(depth_3):
        #     self.flow.append(ActNorm(8))
        #     self.flow.append(InvConv2d(8))
        #     self.flow.append(CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, even], free_net=CNN(c_in=8, c_out=16, n_hidden=128, kernel_size=3, padding=1), c_in=8))
            even = not even
    def forward(self, x, mu, sig):
        sldj = 0
        for i in self.flow:
            x, ldj = i(x)
            sldj += ldj
        x = (x-mu)/torch.exp(sig)
        sldj -= torch.sum(sig)
        sldj = sldj.sum() + self.prior.log_prob(x).sum()
        return sldj
    def sample(self, mu, sig):
        z = self.prior.sample(mu.shape).to(mu.device)
        z = z*torch.exp(sig) + mu
        for i in reversed(self.flow[1:]):
            z, _ = i(z, sample=True)
        return z