import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from parameterized_models import PointwiseLoad

class AugmentedLayer(nn.Module):
    def __init__(self, enc_model, dec_model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enc_model = enc_model
        self.dec_model = dec_model
    def forward(self, x, e, sample=False):
        log_prob = 0
        if sample is False:
            st = self.enc_model(x)
            s, t = st.chunk(2, dim=-1)
            z = torch.exp(s) * e + t
            log_prob += s.sum(-1, keepdim=True)
            st = self.dec_model(z)
            s, t = st.chunk(2, dim=-1)
            x = x * torch.exp(s) + t
            log_prob += s.sum(-1, keepdim=True)
        else:
            st = self.dec_model(e)
            s, t = st.chunk(2, dim=-1)
            x = torch.exp(-s) * (x - t)
            log_prob -= s.sum(-1, keepdim=True)
            st = self.enc_model(x)
            s, t = st.chunk(2, dim=-1)
            z = torch.exp(-s) * (e - t)
            log_prob -= s.sum(-1, keepdim=True)
        return x, z, log_prob
    
class AugmentedFlowParam(nn.Module):
    def __init__(self, num_layers, enc_dec_config):
        super().__init__()
        self.enc_dec_config = enc_dec_config
        self.flow = nn.ModuleList([AugmentedLayer(PointwiseLoad()) for _ in range(num_layers)])