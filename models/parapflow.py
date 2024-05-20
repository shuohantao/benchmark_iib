import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.unet import UNetVanilla
from modules.coupling import CouplingLayerScaleInv
from modules.free_net import CNN, CNN_Linear
from modules.resnet import ResNet
from modules.partitions import checkerBoard
from modules.act_norm import ActNorm
from modules.conv import InvConv2d
from modules.squeeze import SqueezeFlow
from modules.split import SplitFlow

class parapflow(nn.Module):
    def __init__(self, in_d, num_head, depth_1, depth_2, depth_3, device='cuda'):
        super(parapflow, self).__init__()
        self.num_head = num_head
        self.in_d = in_d
        self.unet = UNetVanilla(in_d, num_head*in_d)
        self.prior = torch.distributions.Normal(0, 1)
        even = True
        self.flow = nn.ModuleList()
        for i in range(depth_1):
            self.flow.append(ActNorm(in_d*num_head))
            self.flow.append(CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, even], free_net= ResNet(in_d*num_head, 16, 2*in_d*num_head, 2, False), c_in=in_d*num_head))
            even = not even
        self.flow.append(SqueezeFlow())
        for i in range(depth_2):
            self.flow.append(ActNorm(4*in_d*num_head))
            self.flow.append(InvConv2d(4*in_d*num_head))
            self.flow.append(CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, even], free_net= ResNet(4*in_d*num_head, 32, 8*in_d*num_head, 2, False), c_in=4*in_d*num_head))
            even = not even
        self.flow.append(SplitFlow(prior=self.prior, device=torch.device('cuda')))
        self.flow.append(SqueezeFlow())
        for i in range(depth_3):
            self.flow.append(ActNorm(8*in_d*num_head))
            self.flow.append(InvConv2d(8*in_d*num_head))
            self.flow.append(CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, even], free_net= ResNet(8*in_d*num_head, 64, 16*in_d*num_head, 2, False), c_in=8*in_d*num_head))
            even = not even

    def forward(self, x):
        attention = self.unet(x)
        weight = F.softmax(attention, dim=1)
        x = x * weight
        sldj = 0
        for layer in self.flow:
            x, ldj = layer(x)
            sldj += ldj
        return -sldj.sum() - self.prior.log_prob(x).sum()
    def sample(self, num_samples, device, **kwargs):
        x = self.prior.sample((num_samples, self.num_head*self.in_d*8, 7, 7)).to(device)
        for layer in reversed(self.flow):
            x, _ = layer(x, sample=True)
        x = x.sum(dim=1, keepdim=True)
        return x
