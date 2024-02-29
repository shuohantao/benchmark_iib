from modules.wav_flow import *
from modules.coupling import *
from modules.partitions import checkerBoard
from modules.act_norm import ActNorm
from modules.split import SplitFlow
from modules.squeeze import SqueezeFlow
from modules.conv import InvConv2d
from modules.filters import wavelet_seg
import torch.nn as nn
import torch
from torch.distributions.normal import Normal
from haar_pytorch import HaarInverse

class WaveletFlow(nn.Module):
    def __init__(self, num_scales, max_res, step_settings, conv_settings, base_split):
        super().__init__()
        self.num_scales = num_scales
        self.max_res = max_res
        self.prior = Normal(0, 1)
        self.flows = nn.ModuleList([nn.ModuleList([
                        k for i in range(step_settings[j]) for k in [ActNorm(3), InvConv2d(3), CouplingLayer(partition=checkerBoard([max_res//(2**(j+1))]*2, i%2==0),
                        c_in=3,
                        free_net=WaveFree(4, conv_settings[j], 6))]]) for j in range(num_scales)])
        self.base_split = base_split
        if not base_split:
            self.uncon_flow = nn.ModuleList([
                                        k for i in range(step_settings[-1]) for k in [ActNorm(1), CouplingLayer(partition=checkerBoard([max_res//(2**(num_scales))]*2, i%2==0),
                                        c_in=1,
                                        free_net=WaveFree(1, conv_settings[-1], 2))]])
        else:
            self.uncon_flow = nn.ModuleList([
                                        k for i in range(step_settings[-1]//4) for k in [ActNorm(1), CouplingLayer(partition=checkerBoard([max_res//(2**(num_scales))]*2, i%2==0),
                                        c_in=1,
                                        free_net=WaveFree(1, conv_settings[-1], 2))]])
            self.uncon_flow.append(SqueezeFlow())
            self.uncon_flow.extend([
                                        k for i in range(step_settings[-1]//4) for k in [ActNorm(4), InvConv2d(4), CouplingLayer(partition=checkerBoard([max_res//(2**(num_scales+1))]*2, i%2==0),
                                        c_in=4,
                                        free_net=WaveFree(4, conv_settings[-1], 8))]])
            self.uncon_flow.append(SqueezeFlow())
            self.uncon_flow.append(SplitFlow(self.prior, torch.device('cuda')))
            self.uncon_flow.extend([
                                        k for i in range(step_settings[-1]-step_settings[-1]//2) for k in [ActNorm(8), InvConv2d(8), CouplingLayer(partition=checkerBoard([max_res//(2**(num_scales+2))]*2, i%2==0),
                                        c_in=8,
                                        free_net=WaveFree(8, conv_settings[-1], 16))]])
        self.shape_list = [max_res//(2**(i+1)) for i in range(num_scales)]
        self.ihaar = HaarInverse()
    def forward(self, x):
        segs = wavelet_seg(x, self.num_scales)
        snll = 0
        for i, j in enumerate(self.flows):
            nll = 0
            target = segs[i][1]
            con = segs[i][0]
            for k in j:
                target, nl = k(target, condition=con)
                nll += nl
            snll += nll.sum() + self.prior.log_prob(target).sum()
        nll = 0
        target = segs[-1][0]
        for i in self.uncon_flow:
            target, nl = i(target)
            nll += nl
        snll += nll.sum() + self.prior.log_prob(target).sum()
        return -snll
    def reverse_pass(self, flow, latent, condition=None):
        for i in reversed(flow):
            latent, _ = i(latent, sample=True, condition=condition)
        return latent
    def sample(self, num_samples, resolution, device):
        assert resolution in self.shape_list + [self.max_res], "Resolution not supported"
        if not self.base_split:
            latent = self.prior.sample((num_samples, 1, self.shape_list[-1], self.shape_list[-1])).to(device)
        else:
            latent = self.prior.sample((num_samples, 8, self.shape_list[-1]//4, self.shape_list[-1]//4)).to(device)
        con = self.reverse_pass(self.uncon_flow, latent)
        if con.shape[-1] == resolution:
            return con
        for i, j in enumerate(reversed(self.flows)):
            latent = self.prior.sample((num_samples, 3, self.shape_list[-(i+1)], self.shape_list[-(i+1)])).to(device)
            latent = self.reverse_pass(j, latent, condition=con)
            con = self.ihaar(torch.cat([con, latent], dim=1))
            if con.shape[-1] == resolution:
                break
        return con
                

        
