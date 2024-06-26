import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys; sys.path.append(os.path.abspath(os.path.join('..'))); sys.path.append(os.path.abspath(os.path.join('.')))
from torch.distributions.normal import Normal
from modules.squeeze import SqueezeFlow
from modules.split import SplitFlow
from modules.coupling import CouplingLayerScaleInv
from modules.dequantization import Dequantization
from modules.conv import InvConv2d
from modules.free_net import CNN_Linear, NeuropCNN, GatedConvNet
from modules.filters import frequency_seg
from modules.filters import _perform_dft, _perform_idft
from modules.act_norm import ActNorm
from modules.partitions import checkerBoard
from modules.freq_norm import FreqNorm
import numpy as np
class CAF(nn.Module):
    def __init__(self, shape, depth_1=4, depth_2=4, depth_3=8, modes = 16, prior=Normal(0, 1), normalize=False, fourier=False, **kwargs):
        super().__init__()
        self.prior = prior
        self.flow = nn.ModuleList()
        # self.flow.append(Dequantization())
        self.dummy_squeeze = SqueezeFlow()
        self.shape = shape
        self.modes = modes
        self.normalize = normalize
        self.ar_flow = nn.ModuleList([ActNorm(1),
                                      CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, True], free_net=NeuropCNN(c_in=2, c_hidden=16, default_shape=16, kernel_shapes=[3]*3, fourier=fourier), c_in=1),
                                      ActNorm(1),
                                      CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, False], free_net=NeuropCNN(c_in=2, c_hidden=16, default_shape=16, kernel_shapes=[3]*3, fourier=fourier), c_in=1),
                                      ActNorm(1),
                                      CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, True], free_net=NeuropCNN(c_in=2, c_hidden=16, default_shape=16, kernel_shapes=[3]*3, fourier=fourier), c_in=1),
                                      ActNorm(1),
                                      CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, False], free_net=NeuropCNN(c_in=2, c_hidden=32, default_shape=16, kernel_shapes=[3]*3, fourier=fourier), c_in=1),
                                      ActNorm(1),
                                      CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, True], free_net=NeuropCNN(c_in=2, c_hidden=32, default_shape=16, kernel_shapes=[3]*3, fourier=fourier), c_in=1),
                                      ActNorm(1),
                                      CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, False], free_net=NeuropCNN(c_in=2, c_hidden=32, default_shape=16, kernel_shapes=[3]*3, fourier=fourier), c_in=1),
                                      SqueezeFlow(),
                                      SplitFlow(self.prior, torch.device('cuda'), test_inv=False),
                                      ActNorm(2),
                                      InvConv2d(2),
                                      CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, True], free_net=NeuropCNN(c_in=6, c_out=4, c_hidden=64, default_shape=8, fourier=fourier), c_in=2),
                                      ActNorm(2),
                                      InvConv2d(2),
                                      CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, False], free_net=NeuropCNN(c_in=6, c_out=4, c_hidden=64, default_shape=8, fourier=fourier), c_in=2),
                                      ActNorm(2),
                                      InvConv2d(2),
                                      CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, True], free_net=NeuropCNN(c_in=6, c_out=4, c_hidden=128, default_shape=8, fourier=fourier), c_in=2),
                                      ActNorm(2),
                                      InvConv2d(2),
                                      CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, False], free_net=NeuropCNN(c_in=6, c_out=4, c_hidden=128, default_shape=8, fourier=fourier), c_in=2),
                                      ActNorm(2),
                                      InvConv2d(2),
                                      CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, True], free_net=NeuropCNN(c_in=6, c_out=4, c_hidden=128, default_shape=8, fourier=fourier), c_in=2),
                                      ActNorm(2),
                                      InvConv2d(2),
                                      CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, False], free_net=NeuropCNN(c_in=6, c_out=4, c_hidden=128, default_shape=8, fourier=fourier), c_in=2),
                                      ])
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
        # sldj = 0
        # x, ldj = self.flow[0](x)
        # deq_ldj = ldj.sum()
        sldj = 0
        x, seg_list = frequency_seg(x, self.modes, 1, torch.device('cuda'))
        # x.requires_grad = True ##uncomment for normal dequant
        for i in self.flow:
            x, ldj = i(x)
            sldj += ldj
        base_sldj = sldj.sum() + self.prior.log_prob(x).sum()
        ar_sldj = 0
        for k, i in enumerate(seg_list):
            con = i[1]
            x = i[0]
            sldj = 0
            for j in self.ar_flow:
                if isinstance(j, ActNorm) or isinstance(j, SqueezeFlow) or isinstance(j, SplitFlow) or isinstance(j, InvConv2d):
                    x, ldj = j(x)
                    if isinstance(j, SqueezeFlow):
                        con, _ = self.dummy_squeeze(con)
                else:
                    x, ldj = j(x, condition=con)
                sldj += ldj
            ar_sldj += sldj.sum() + self.prior.log_prob(x).sum()
        return -base_sldj-ar_sldj
    def sample(self, num_samples, resolution,  device, a=2, lm=1, eps=0.1, **kwargs):
        with torch.no_grad():
            prior_shape = self.modes
            z = self.prior.sample(torch.Size([num_samples]+[self.shape[-3]*8, prior_shape//4, prior_shape//4])).to(device)
            for i in reversed(self.flow):
                z, _ = i(z, sample=True)
            img = z
            con = _perform_dft(img)
            con = F.pad(con, (1, 1, 1, 1), 'constant', 0)
            con = _perform_idft(con)
            con, _ = self.dummy_squeeze(con)
            for i in range((resolution-prior_shape)//2):
                prior_shape = torch.Size([num_samples, 2, con.shape[-2], con.shape[-1]])
                sub_img = self.prior.sample(prior_shape).to(device)
                for j in reversed(self.ar_flow):
                    if isinstance(j, ActNorm) or isinstance(j, SqueezeFlow) or isinstance(j, SplitFlow) or isinstance(j, InvConv2d):
                        sub_img, _ = j(sub_img, sample=True)
                        if isinstance(j, SqueezeFlow):
                            con, _ = self.dummy_squeeze(con, sample=True)
                    else:
                        sub_img, _ = j(sub_img, sample=True, condition=con)

                if self.normalize:
                    sub_img *= ((1-eps)*a**(-lm*i) + eps)
                img = sub_img + con
                con = _perform_dft(sub_img+con)
                con = F.pad(con, (1, 1, 1, 1), 'constant', 0)
                con = _perform_idft(con)
                con, _ = self.dummy_squeeze(con)
            # img, _ = self.flow[0](img, sample=True)
            assert not torch.isnan(img).any()
            return img