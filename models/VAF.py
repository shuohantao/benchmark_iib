import torch
import torch.nn as nn
import torch.optim as optim

class VAF(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        encoder = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.ar = nn.TransformerEncoder(encoder_layer=encoder, num_layers=4)
        self.upper = 128
        self.f0 = nn.Sequential(nn.Linear(self.upper, self.upper), nn.ReLU(), nn.Linear(256, 128))
    def forward(self, x):
        x_f = torch.fft.fft2(x, s=(x.shape[-1], x.shape[-2]), dim=(-1, -2), norm='ortho')
        x_f = x_f[..., :, :x.shape[-1]//2]
        x_r = x_f.real
        x_i = x_f.imag
        x_f = torch.cat([x_r, x_i], dim=1)
        # Create positional encoding for pixels in x_r using both x and y coordinates
        batch_size, _, height, width = x_r.size()
        pos_enc = torch.zeros((batch_size, 4, height, width), dtype=torch.float32, device=x_r.device)

        x_coords = torch.arange(width, dtype=torch.float32, device=x_r.device).view(1, 1, 1, -1)
        y_coords = torch.arange(height, dtype=torch.float32, device=x_r.device).view(1, 1, -1, 1)

        pos_enc[:, 0, :, :] = torch.sin(x_coords / 10000)
        pos_enc[:, 1, :, :] = torch.cos(x_coords / 10000)
        pos_enc[:, 2, :, :] = torch.sin(y_coords / 10000)
        pos_enc[:, 3, :, :] = torch.cos(y_coords / 10000)

        x_r_with_pos_enc = torch.cat([x_f, pos_enc], dim=1)

    

