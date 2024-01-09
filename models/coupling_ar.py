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
from modules.free_net import CNN_Linear, NeuropCNNConditioned, NeuropCNN
from modules.filters import frequency_seg
from modules.filters import _perform_dft, _perform_idft
from modules.act_norm import ActNorm
from modules.partitions import checkerBoard
from modules.freq_norm import FreqNorm
class HelperManager():
    def __init__(self, n_iter):
        self.n_iter = n_iter
    def get_config(self):
        return {'n_spectral_iter':self.n_iter}
class CouplingFlowAR(nn.Module):
    def __init__(self, shape, depth_1=4, depth_2=4, depth_3=8, n_iter = 50, modes = 16, prior=Normal(0, 1), fourier=False, normalize=False):
        super().__init__()
        self.prior = prior
        self.flow = nn.ModuleList()
        ################# Dequantization ###################
        self.flow.append(Dequantization())
        ####################################################
        self.dummy_squeeze = SqueezeFlow()
        self.shape = torch.Size(shape)
        self.n_iter_m = HelperManager(n_iter=n_iter)
        self.modes = modes
        self.ar_flow = nn.ModuleList()
        for i in range(2):
            self.ar_flow.extend([ActNorm(1),
                        #   CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, i%2==0], free_net=NeuropCNNConditioned(c_in=2, c_hidden=128, default_shape=modes, fourier=fourier), c_in=1)])
                                 CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, i%2==0], free_net=NeuropCNN(c_in=2, c_hidden=64, default_shape=modes, kernel_shapes=[7]*3, fourier=fourier), c_in=1)])
        for i in range(2):
            self.ar_flow.extend([ActNorm(1),
                        #   CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, i%2==0], free_net=NeuropCNNConditioned(c_in=2, c_hidden=128, default_shape=modes, fourier=fourier), c_in=1)])
                                 CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, i%2==0], free_net=NeuropCNN(c_in=2, c_hidden=128, default_shape=modes, kernel_shapes=[5]*3, fourier=fourier), c_in=1)])
        for i in range(2):
            self.ar_flow.extend([ActNorm(1),
                        #   CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, i%2==0], free_net=NeuropCNNConditioned(c_in=2, c_hidden=128, default_shape=modes, fourier=fourier), c_in=1)])
                                 CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, i%2==0], free_net=NeuropCNN(c_in=2, c_hidden=256, default_shape=modes, kernel_shapes=[3]*3, fourier=fourier), c_in=1)])
        self.ar_flow.append(SqueezeFlow())
        self.ar_flow.append(SplitFlow(self.prior, torch.device('cuda')))
        for i in range(8):
            self.ar_flow.extend([ActNorm(2),
                          InvConv2d(2),
                          CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, i%2==1], free_net=NeuropCNN(c_in=6, c_out=4, c_hidden=256, default_shape=modes//2, fourier=fourier), c_in=2)])
        for i in range(depth_1):
            self.flow.append(ActNorm(1))
            self.flow.append(CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, i%2==0], free_net= CNN_Linear(c_in=1, c_out=2, c_hidden=32, shape=torch.Size((1, 1, modes, modes))), c_in=1))
        self.flow.append(SqueezeFlow())
        for i in range(depth_2):
            self.flow.append(ActNorm(4))
            self.flow.append(InvConv2d(4))
            self.flow.append(CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, i%2==1], free_net=CNN_Linear(c_in=4, c_out=8, c_hidden=64, shape=torch.Size((1, 4, modes//2, modes//2))), c_in=4))
        self.flow.append(SplitFlow(prior=prior, device=torch.device('cuda')))
        self.flow.append(SqueezeFlow())
        for i in range(depth_3):
            self.flow.append(ActNorm(8))
            self.flow.append(InvConv2d(8))
            self.flow.append(CouplingLayerScaleInv(device=torch.device('cuda'), partition_even=[checkerBoard, i%2==0], free_net=CNN_Linear(c_in=8, c_out=16, c_hidden=128, shape=torch.Size((1, 8, modes//4, modes//4))), c_in=8))
        self.freq_norm = None
        if normalize:
            self.freq_norm = FreqNorm(n_hid=32)
    def forward(self, x):
        init_shape = x.shape[-1]
        x, ldj = self.flow[0](x)
        deq_ldj = ldj.sum()
        sldj = 0
        x, seg_list = frequency_seg(x, self.modes, 1, torch.device('cuda'))
        for i in self.flow[1:]:
            x, ldj = i(x)
            sldj += ldj.sum()
        base_sldj = sldj + self.prior.log_prob(x).sum()
        ar_sldj = 0
        if init_shape > self.modes:
            for k, i in enumerate(seg_list):
                con = i[1]
                x = i[0]
                sldj = 0
                if self.freq_norm is not None:
                    x, ldj = self.freq_norm(x)
                    sldj += ldj.sum()
                for j in self.ar_flow:
                    if isinstance(j, ActNorm) or isinstance(j, SqueezeFlow) or isinstance(j, SplitFlow) or isinstance(j, InvConv2d):
                        x, ldj = j(x)
                        if isinstance(j, SqueezeFlow):
                            con, _ = self.dummy_squeeze(con)
                    else:
                        # x, ldj = j(x, condition=con, neurop_condition=k)
                        x, ldj = j(x, condition=con)
                    sldj += ldj.sum()
                ar_sldj += sldj + self.prior.log_prob(x).sum()
        return -deq_ldj-base_sldj-ar_sldj
    def sample(self, resolution, num_samples=1, device="cuda", **kwargs):
        prior_shape = self.modes
        assert prior_shape <= resolution, "Prior shape must be smaller than resolution"
        with torch.no_grad():
            z = self.prior.sample(torch.Size([num_samples]+[self.shape[-3]*8, prior_shape//4, prior_shape//4])).to(device)
            for i in reversed(self.flow[1:]):
                if isinstance(i, ActNorm):
                    z, _ = i(z, move_towards_base=False)
                else:
                    z, _ = i(z, sample=True)
            img = z
            if self.modes < resolution:
                con = _perform_dft(img)
                con = F.pad(con, (1, 1, 1, 1), 'constant', 0)
                con = _perform_idft(con)
                con, _ = self.dummy_squeeze(con)
                for i in reversed(range((resolution-prior_shape)//2)):
                    prior_shape = torch.Size([num_samples, 2, con.shape[-2], con.shape[-1]])
                    sub_img = self.prior.sample(prior_shape).to(device)
                    for j in reversed(self.ar_flow):
                        if isinstance(j, ActNorm):
                            sub_img, _ = j(sub_img, move_towards_base=False)
                        elif isinstance(j, SqueezeFlow) or isinstance(j, SplitFlow) or isinstance(j, InvConv2d):
                            sub_img, _ = j(sub_img, sample=True)
                            if isinstance(j, SqueezeFlow):
                                con, _ = self.dummy_squeeze(con, sample=True)
                        else:
                            sub_img, _ = j(sub_img, sample=True, condition=con)
                    if self.freq_norm is not None:
                        sub_img, _ = self.freq_norm(sub_img, sample=True)
                    img = sub_img + con
                    con = _perform_dft(sub_img+con)
                    con = F.pad(con, (1, 1, 1, 1), 'constant', 0)
                    con = _perform_idft(con)
                    con, _ = self.dummy_squeeze(con)
            img, _ = self.flow[0](img, sample=True)
            return img