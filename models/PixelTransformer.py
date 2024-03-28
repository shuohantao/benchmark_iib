import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class PixelTransformer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        encoder = nn.TransformerEncoderLayer(d_model=5, nhead=5, dim_feedforward=2048, dropout=0.1)
        self.ar = nn.TransformerEncoder(encoder_layer=encoder, num_layers=8)
        self.f0 = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 32))
        self.prior = Normal(0, 1)
        self.scaling_factor = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        batch_size, _, height, width = x.size()
        pos_enc = torch.zeros((batch_size, 4, height, width), dtype=torch.float32, device=x.device)
        window = height * 2
        x_coords = torch.arange(width, dtype=torch.float32, device=x.device).view(1, 1, 1, -1)
        y_coords = torch.arange(height, dtype=torch.float32, device=x.device).view(1, 1, -1, 1)

        pos_enc[:, 0, :, :] = torch.sin(x_coords / 10000)
        pos_enc[:, 1, :, :] = torch.cos(x_coords / 10000)
        pos_enc[:, 2, :, :] = torch.sin(y_coords / 10000)
        pos_enc[:, 3, :, :] = torch.cos(y_coords / 10000)
        start_token = torch.zeros((batch_size, 1, 5), dtype=torch.float32, device=x.device)
        x = torch.cat([x, pos_enc], dim=1)
        x = x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1)
        x = torch.cat([start_token, x], dim=1)
        sldj = 0
        logits = []
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        for i in range(x.shape[-2]-1):
            masked = x.clone()[:, max(0, i+2-window):i+2, :]
            masked[:, -1, 0] = -1
            st = self.ar(masked)
            st = st[:, -1, :]
            st = self.f0(st)
            st = st.view(-1, 2, 16)
            s, t = st.chunk(2, dim=1)
            s = s.split(1, dim=-1)
            t = t.split(1, dim=-1)
            z = x[:, i+1, 0].clone()
            ldj = 0
            for j in range(16):
                scale = s[j].view(-1)
                scale = torch.tanh(scale / s_fac) * s_fac
                shift = t[j].view(-1)
                z = torch.exp(scale) * z + shift
                ldj += s[j]
            logits.append(z.unsqueeze(-1))
            sldj += ldj.sum()
        z = torch.cat(logits, dim=-1)
        sldj += self.prior.log_prob(z).sum()
        return -sldj
    def sample(self, num_samples, resolution,  device):
        self.ar.eval()
        start_token = torch.zeros((num_samples, 1, 5), dtype=torch.float32, device=device)
        pos_enc = torch.zeros((num_samples, 4, resolution, resolution), dtype=torch.float32, device=device)
        window = 2*resolution
        x_coords = torch.arange(resolution, dtype=torch.float32, device=device).view(1, 1, 1, -1)
        y_coords = torch.arange(resolution, dtype=torch.float32, device=device).view(1, 1, -1, 1)

        pos_enc[:, 0, :, :] = torch.sin(x_coords / 10000)
        pos_enc[:, 1, :, :] = torch.cos(x_coords / 10000)
        pos_enc[:, 2, :, :] = torch.sin(y_coords / 10000)
        pos_enc[:, 3, :, :] = torch.cos(y_coords / 10000)
        x = -torch.ones((num_samples, 1, resolution, resolution), dtype=torch.float32, device=device)
        x = torch.cat([x, pos_enc], dim=1)
        x = x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1)
        x = torch.cat([start_token, x], dim=1)
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        for i in range(x.shape[-2]-1):
            masked = x[:, max(0, i+2-window):i+2, :]
            st = self.ar(masked)
            st = st[:, -1, :]
            st = self.f0(st)
            st = st.view(-1, 2, 16)
            s, t = st.chunk(2, dim=1)
            s = s.split(1, dim=-1)
            t = t.split(1, dim=-1)
            z = self.prior.sample(torch.Size((num_samples,))).to(device).view(-1)
            for j in range(16):
                scale = s[-j-1].view(-1)
                scale = torch.tanh(scale / s_fac) * s_fac
                shift = t[-j-1].view(-1)
                z = (z - shift) / torch.exp(scale)
            x[:, i+1, 0] = z
        return x[:, 1:, 0].view(num_samples, 1, resolution, resolution)
            