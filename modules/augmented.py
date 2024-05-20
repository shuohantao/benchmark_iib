import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from modules.param_model import PointwiseLoad
from modules.act_norm import ActNormAug
class AugmentedLayerParam(nn.Module):
    def __init__(self, enc_model, dec_model, c_in=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enc_model = enc_model
        self.dec_model = dec_model
        # self.enc_scale = nn.Parameter(torch.zeros(c_in))
        # self.dec_scale = nn.Parameter(torch.zeros(c_in))
    def forward(self, x, e, param, sample=False):
        log_prob = 0
        enc_param, dec_param = param[..., :-2].chunk(2, dim=-1) 
        enc_scale, dec_scale = param[..., -2:].chunk(2, dim=-1)
        if sample is False:
            st = self.enc_model(x, enc_param)
            s, t = st.chunk(2, dim=-1)
            s = torch.tanh(s / enc_scale.exp()) * enc_scale.exp()
            z = torch.exp(s) * e + t
            log_prob += s.sum(-1, keepdim=True)
            st = self.dec_model(z, dec_param)
            s, t = st.chunk(2, dim=-1)
            s = torch.tanh(s / dec_scale.exp()) * dec_scale.exp()
            x = x * torch.exp(s) + t
            log_prob += s.sum(-1, keepdim=True)
        else:
            st = self.dec_model(e, dec_param)
            s, t = st.chunk(2, dim=-1)
            s = torch.tanh(s / enc_scale.exp()) * enc_scale.exp()
            x = torch.exp(-s) * (x - t)
            log_prob -= s.sum(-1, keepdim=True)
            st = self.enc_model(x, enc_param)
            s, t = st.chunk(2, dim=-1)
            s = torch.tanh(s / dec_scale.exp()) * dec_scale.exp()
            z = torch.exp(-s) * (e - t)
            log_prob -= s.sum(-1, keepdim=True)
        return x, z, log_prob
    
class AugmentedParamFlow(nn.Module):
    def __init__(self, dim_hid, num_layers, dim_in, enc_dec_config=None):
        super().__init__()
        self.enc_dec_config = enc_dec_config
        self.flows = nn.ModuleList([i for _ in range(num_layers) for i in [ActNormAug(dim_in), AugmentedLayerParam(PointwiseLoad(dim_hid, 3, dim_in, dim_in * 2), PointwiseLoad(dim_hid, 3, dim_in, dim_in * 2))]])
        self.num_layers = num_layers
        self.prior = torch.distributions.Normal(0, 1)
    def forward(self, param, x):
        params = param.chunk(self.num_layers, dim=-1)
        z = self.prior.sample(x.shape).to(x.device)
        ldj = 0
        for i, flow in enumerate(self.flows):
            x, z, log_prob = flow(x, z, param=params[i//2])
            ldj += log_prob
        return ldj.sum() + self.prior.log_prob(z).sum() + self.prior.log_prob(x).sum()
    def sample(self, param):
        params = param.chunk(self.num_layers, dim=-1)
        z = self.prior.sample((*param.shape[:-1], 1)).to(param.device)
        x = self.prior.sample((*param.shape[:-1], 1)).to(param.device)
        for i, flow in enumerate(reversed(self.flows)):
            x, z, _ = flow(x, z, param=params[-(i//2+1)], sample=True)
        return x
    def calc_param(self):
        return self.num_layers * 2 * self.flows[1].enc_model.calc_param()
# if __name__ == '__main__':
#     model = AugmentedParamFlow(8)
#     print(model.calc_param())
    
