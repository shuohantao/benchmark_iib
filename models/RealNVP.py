import torch.nn as nn
import torch
from modules.act_norm import ActNorm
from modules.coupling import CouplingLayerScaleInv
from modules.split import SplitFlow
from modules.squeeze import SqueezeFlow
from modules.conv import InvConv2d
from modules.free_net import CNN_Linear, CNN, GatedConvNet
from modules.dequantization import Dequantization
from modules.partitions import checkerBoard

class RealNVP(nn.Module):
    def __init__(self, depth_1=6, depth_2=6, depth_3=12):
        super(RealNVP, self).__init__()
        self.flow = nn.ModuleList()
        # self.flow.append(Dequantization())
        self.prior = torch.distributions.Normal(0, 1)
        even = True
        for i in range(depth_1):
            self.flow.append(ActNorm(1))
            self.flow.append(CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, even], free_net= GatedConvNet(1, 32), c_in=1))
            even = not even
        self.flow.append(SqueezeFlow())
        for i in range(depth_2):
            self.flow.append(ActNorm(4))
            self.flow.append(InvConv2d(4))
            self.flow.append(CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, even], free_net=GatedConvNet(4, 64), c_in=4))
            even = not even
        self.flow.append(SplitFlow(prior=self.prior, device=torch.device('cuda')))
        self.flow.append(SqueezeFlow())
        for i in range(depth_3):
            self.flow.append(ActNorm(8))
            self.flow.append(InvConv2d(8))
            self.flow.append(CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, even], free_net=GatedConvNet(8, 128), c_in=8))
            even = not even
    def forward(self, x):
        sldj = 0
        for i in self.flow:
            x, ldj = i(x)
            sldj += ldj
        nll_loss = sldj.sum() + self.prior.log_prob(x).sum()
        return -nll_loss
    def sample(self, num_samples, resolution, device, **kwargs):
        with torch.no_grad():
            sample_shape = torch.Size([num_samples, 1, resolution, resolution])
            sample_shape = list(sample_shape)
            sample_shape[-3] *= 8
            sample_shape[-2] //=4
            sample_shape[-1] //= 4
            z = self.prior.sample(sample_shape).to(device)
            for i in reversed(self.flow):
                z, _ = i(z, sample=True)
            assert z.isnan().sum() == 0, f"NaN in samples: {z.isnan().sum()}"
            return z