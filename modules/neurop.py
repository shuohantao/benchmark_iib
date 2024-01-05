import torch
import torch.nn as nn
import math
class SineActivation(torch.nn.Module):
    def __init__(self, omega):
        super().__init__()
        self.omega = omega
    def forward(self, x):
        return torch.sin(x*self.omega)
class FourierMapping(torch.nn.Module):
    def __init__(self, sample_var = 10, out_dim=256, in_dim=2) -> None:
        super().__init__()
        self.register_buffer('B', torch.randn(in_dim, out_dim//2) * sample_var)
    def forward(self, x):
        return torch.cat((torch.cos(2 * math.pi * torch.matmul(x, self.B)), torch.sin(2 * math.pi * torch.matmul(x, self.B))), -1)
class NeurOpKrnMLP(nn.Module):
    def __init__(self, c_in, c_out, coord_in = 2, omega=3, conditioned=False):
        super().__init__()
        self.coord_in = coord_in
        self.c_in = c_in
        self.c_out = c_out
        coord_in = coord_in + 1 if conditioned else coord_in
        self.operators = nn.Sequential(nn.Linear(coord_in, 256*coord_in),
                                       SineActivation(omega=omega),
                                       nn.Linear(256*coord_in, 256*coord_in),
                                       SineActivation(omega=omega),
                                       nn.Linear(256*coord_in, 256*coord_in),
                                       SineActivation(omega=omega),
                                       nn.Linear(256*coord_in, c_in*c_out),)
    def forward(self, coord, k_shape, condition=None):
        if condition is not None:
            condition = torch.ones_like(coord[:,:1]) * condition
            coord = torch.cat((coord, condition), -1)
        output = self.operators(coord)
        return output.view(self.c_out, self.c_in, k_shape, k_shape)
        # return output.view(k_shape, k_shape, self.c_out, self.c_in).permute(2, 3, 0, 1)
class NeurOpKrnMLPFF(nn.Module):
    def __init__(self, c_in, c_out, coord_in = 2, ff_dim=128, sample_var=10, conditioned=False):
        super().__init__()
        self.coord_in = coord_in
        self.c_in = c_in
        self.c_out = c_out
        self.ff_map = FourierMapping(sample_var=sample_var, out_dim=ff_dim, in_dim=coord_in)
        in_dim = ff_dim + 1 if conditioned else ff_dim
        self.operators = nn.Sequential(nn.Linear(in_dim, 2*ff_dim),
                                       nn.Softplus(),
                                       nn.Linear(2*ff_dim, 2*ff_dim),
                                       nn.Softplus(),
                                       nn.Linear(2*ff_dim, 2*ff_dim),
                                       nn.Softplus(),
                                       nn.Linear(2*ff_dim, c_in*c_out),)
    def forward(self, coord, k_shape, condition=None):
        coord = self.ff_map(coord)
        if condition is not None:
            condition = torch.ones_like(coord[:,:1]) * condition
            coord = torch.cat((coord, condition), -1)
        output = self.operators(coord)
        return output.view(self.c_out, self.c_in, k_shape, k_shape)
        # return output.view(k_shape, k_shape, self.c_out, self.c_in).permute(2, 3, 0, 1)
import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=13000):
        super(PositionalEncoding, self).__init__()
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x



class NeurOpKrnTfm(nn.Module):
    def __init__(self, input_dim, output_dim, nhead = 1, nhid = 16, nlayers=6, dropout=0.5, omega=10):
        super(NeurOpKrnTfm, self).__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(input_dim)
        encoder_layers = nn.TransformerEncoderLayer(input_dim, nhead, nhid, dropout, activation=SineActivation())
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.act = SineActivation(omega=omega)
        self.decoder = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.act(output)
        output = self.decoder(output)
        return output
