import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys; sys.path.append(os.path.abspath(os.path.join('..'))); sys.path.append(os.path.abspath(os.path.join('.')))
from torch.distributions.normal import Normal
from modules.squeeze import SqueezeFlow
from modules.split import SplitFlow
from modules.coupling import CouplingLayerAlias, CouplingLayerScaleInv
from modules.dequantization import Dequantization
from modules.conv import InvConv2d
from modules.free_net import CNN_Linear, AliasFreeCNN
from modules.filters import frequency_seg
from modules.filters import _perform_dft, _perform_idft
from modules.act_norm import ActNorm
from modules.partitions import checkerBoard
from modules.freq_norm import FreqNorm
from modules.grf import GRF
import numpy as np
class CAFGRF(nn.Module):
    def __init__(self, shape, depth_1=4, depth_2=4, depth_3=8, ar_depth=8, ar_hid=64, num_res=3, modes = 16, prior=Normal(0, 1), **kwargs):
        super().__init__()
        self.prior = prior
        self.grf = GRF(0.5, 5)
        self.flow = nn.ModuleList()
        self.flow.append(Dequantization())
        self.dummy_squeeze = SqueezeFlow()
        self.shape = shape
        self.modes = modes
        self.ar_flow = nn.ModuleList([k for i in range(ar_depth) for k in [ActNorm(1),
                                                                           CouplingLayerAlias(i%2==0, device=torch.device('cuda'), free_net=AliasFreeCNN(in_c=2, out_c=2, num_res=num_res, hid_c=ar_hid), c_in=1)]])
        even = True
        for i in range(depth_1):
            self.flow.append(ActNorm(1))
            self.flow.append(CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, even], free_net= CNN_Linear(c_in=1, c_out=2, c_hidden=16, shape=torch.Size((1, 1, modes, modes))), c_in=1))
            even = not even
        self.flow.append(SqueezeFlow())
        for i in range(depth_2):
            self.flow.append(ActNorm(4))
            self.flow.append(InvConv2d(4))
            self.flow.append(CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, even], free_net=CNN_Linear(c_in=4, c_out=8, c_hidden=32, shape=torch.Size((1, 4, modes//2, modes//2))), c_in=4))
            even = not even
        self.flow.append(SplitFlow(prior=prior, device=torch.device('cuda')))
        self.flow.append(SqueezeFlow())
        for i in range(depth_3):
            self.flow.append(ActNorm(8))
            self.flow.append(InvConv2d(8))
            self.flow.append(CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, even], free_net=CNN_Linear(c_in=8, c_out=16, c_hidden=64, shape=torch.Size((1, 8, modes//4, modes//4))), c_in=8))
            even = not even
    def forward(self, x):
        x, ldj = self.flow[0](x)
        deq_ldj = ldj.sum()
        base_sldj = 0
        x, seg_list = frequency_seg(x, self.modes, 1, torch.device('cuda'))
        # x.requires_grad = True ##uncomment for normal dequant
        for i in self.flow[1:]:
            x, ldj = i(x)
            base_sldj += ldj
        base_sldj = base_sldj.sum()
        base_sldj += self.prior.log_prob(x).sum()
        ar_sldj = 0
        latent = 0
        for k, i in enumerate(seg_list):
            sldj = 0
            con = i[1]
            x = i[0]
            for j in self.ar_flow:
                x, ldj = j(x, condition=con)
                sldj += ldj
            ar_sldj += sldj.sum()
            latent += x
            if k != len(seg_list)-1:
                latent = _perform_dft(latent)
                latent = F.pad(latent, (1, 1, 1, 1), 'constant', 0)
                latent = _perform_idft(latent)
        prior_ldj, grf_ldj = self.grf.calc_likelihood(latent, self.modes)
        grf_ldj = grf_ldj.sum()
        return -deq_ldj-base_sldj-ar_sldj-grf_ldj-prior_ldj
    def sample(self, num_samples, resolution,  device, **kwargs):
        with torch.no_grad():
            prior_shape = self.modes
            scales = [self.modes + 2*i for i in range((resolution-self.modes)//2 + 1)]
            latents = self.grf.sample_scales(scales=scales, num=num_samples, device = device)
            z = self.prior.sample(torch.Size([num_samples]+[self.shape[-3]*8, prior_shape//4, prior_shape//4])).to(device)
            for i in reversed(self.flow[1:]):
                z, _ = i(z, sample=True)
            img = z
            for i in range((resolution-prior_shape)//2):
                con = _perform_dft(img)
                con = F.pad(con, (1, 1, 1, 1), 'constant', 0)
                con = _perform_idft(con)
                sub_img = latents[i+1]
                for j in reversed(self.ar_flow):
                    sub_img, _ = j(sub_img, sample=True, condition=con)
                img = sub_img + con
            img, _ = self.flow[0](img, sample=True)
            return img