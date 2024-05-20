import torch.nn as nn
import torch
from modules.fno import FNO2d, FNO2dv2
from modules.free_net import CNN_Linear

class CouplingLayer(nn.Module):
    def __init__(self, partition, free_net, c_in, device=torch.device('cuda')):
        super().__init__()
        self.free_net = free_net
        self.scaling_factor = nn.Parameter(torch.zeros(c_in))
        self.epsilon = 1e-5
        self.partition = partition
        self.device = device
    def forward(self, x, sample=False, condition=None):
        mask = self.partition.to(self.device)
        x_left = x * mask
        if condition is not None:
            st = self.free_net(torch.cat([x_left, condition], dim=1))
        else:
            st = self.free_net(x_left)
        s, t = st.chunk(2, dim=1)
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        s = torch.tanh(s / s_fac) * s_fac
        s = s * (1 - mask)
        t = t * (1 - mask)
        if not sample:
            z = x * torch.exp(s) + t
            ldj = s.sum(dim=[1, 2, 3])
        else:
            z = (x - t) * torch.exp(-s)
            ldj = - s.sum(dim=[1, 2, 3])
        return z, ldj

class CouplingLayerScaleInv(nn.Module):
    def __init__(self, partition_even, device, free_net, c_in, use_unet=False):
        super().__init__()
        self.free_net = free_net
        self.use_unet=use_unet
        self.scaling_factor = nn.Parameter(torch.randn(c_in)/c_in**0.5)
        self.shift_factor = nn.Parameter(torch.randn(c_in)/c_in**0.5)
        self.epsilon = 1e-5
        self.partition = partition_even[0]
        self.even = partition_even[1]
        self.device = device
    def forward(self, x, sample=False, condition=None, mask=None, **kwargs):
        shape = x.shape
        mask = self.partition(shape, self.even).to(self.device)
        x_left = x * mask
        if condition is not None:
            st = self.free_net(torch.cat([x_left, condition], dim=1), shape=shape)
            s, t = st.chunk(2, dim=1)
        else:
            st = self.free_net(x_left, shape=shape)
            s, t = st.chunk(2, dim=1)
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        t_fac = self.shift_factor.exp().view(1, -1, 1, 1)
        s = torch.tanh(s / s_fac) * s_fac
        t = torch.tanh(t / t_fac) * t_fac 
        s = s * (1 - mask)
        t = t * (1 - mask)
        if not sample:
            z = x * torch.exp(s) + t
            ldj = s.sum(dim=[1, 2, 3])
        if sample:
            # print(torch.max(s), torch.min(s), torch.max(t), torch.min(t), self.scaling_factor, s.isnan().any())
            z = (x - t) * torch.exp(-s)
            ldj = - s.sum(dim=[1, 2, 3])
        return z, ldj     

class CouplingLayer1D(nn.Module):
    def __init__(self, partition, free_net, device=torch.device('cuda')):
        super().__init__()
        self.free_net = free_net
        self.scaling_factor = nn.Parameter(torch.zeros(1))
        self.epsilon = 1e-5
        self.partition = partition
        self.device = device
    def forward(self, x, sample=False, condition=None):
        mask = self.partition.to(self.device)
        x_left = x * mask
        if condition is not None:
            st = self.free_net(torch.cat([x_left, condition], dim=1))
        else:
            st = self.free_net(x_left)
        s, t = st.chunk(2, dim=-1)
        s_fac = self.scaling_factor.exp().view(1, 1)
        s = torch.tanh(s / s_fac) * s_fac
        s = s * (1 - mask)
        t = t * (1 - mask)
        if not sample:
            z = x * torch.exp(s) + t
            ldj = s.sum(dim=-1)
        else:
            z = (x - t) * torch.exp(-s)
            ldj = - s.sum(dim=-1)
        return z, ldj 
    
class CouplingLayerFNO(nn.Module):
    def __init__(self, even, c_in, modes, net_cin, net_cout, net_width):
        super().__init__()
        self.scaling_factor = nn.Parameter(torch.zeros(c_in, dtype=torch.cfloat))
        self.epsilon = 1e-5
        self.even = even
        self.net = FNO2d(modes, modes//2, net_width, net_cin, net_cout)
    def forward(self, x, r, sample=False, condition=None):
        x_left, x_right = self.partition(x)
        x_unc = x_left if self.even else x_right
        x_ch = x_right if self.even else x_left
        if condition is not None:
            st = self.net(torch.cat([x_unc, condition], dim=1), r)
        else:
            st = self.net(x_unc, r)
        st_left, st_right = self.partition(st)
        st = st_right if self.even else st_left
        st = torch.fft.rfft2(st, dim=(-2, -1), norm='ortho')
        s, t = st.chunk(2, dim=1)
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        s = torch.tanh(s / s_fac) * s_fac
        x_ch = torch.fft.rfft2(x_ch, dim=(-2, -1), norm='ortho')
        if not sample:
            z = x_ch * torch.exp(s) + t
            ldj = torch.abs(s).sum(dim=[1, 2, 3])*2
            z = torch.fft.irfft2(z, s=(x.size(-2), x.size(-1)), norm='ortho')
        else:
            z = (x_ch - t) * torch.exp(-s)
            ldj = - torch.abs(s).sum(dim=[1, 2, 3])*2
            z = torch.fft.irfft2(z, s=(x.size(-2), x.size(-1)), norm='ortho')
        return z + x_unc, ldj
    def partition(self, x, space=False):
        x_f = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        size = x.shape[-2]
        x_left = x_f.clone()
        x_right = x_f.clone()
        x_left[..., size//2:, :] = 0
        x_right[..., :size//2, :] = 0
        if space:
            x_left = torch.fft.irfft2(x_left, s=(x.size(-2), x.size(-1)), norm='ortho')
            x_right = torch.fft.irfft2(x_right, s=(x.size(-2), x.size(-1)), norm='ortho')
        return x_left, x_right
    
class CouplingLayerFNOv2(nn.Module):
    def __init__(self, shape, even, c_in, modes, net_cin, net_cout, net_width, norm_coeff = 1):
        super().__init__()
        self.scaling_factor = nn.Parameter(torch.ones(c_in))
        self.epsilon = 1e-5
        self.even = even
        #self.net = FNO2dv2(modes, modes//2, net_width, net_cin, net_cout)
        self.net = CNN_Linear(c_in=1, shape=shape, c_out=2)
        self.norm_coeff = norm_coeff
    def forward(self, x, sample=False, condition=None):
        h, w = x.shape[-2:]
        mask = torch.ones_like(x).to(x.device)
        mask[..., :h//2, :w//2] = 0
        mask[..., -h//2:, -w//2:] = 0
        x_left, x_right, x_f = self.partition(x, mask)
        x_unc = x_left if self.even else x_right
        if condition is not None:
            st = self.net(torch.cat([x_unc, condition], dim=1))
        else:
            st = self.net(x_unc)
        st = torch.fft.fft2(st, dim=(-2, -1), s=(h, w), norm='ortho')
        s, t = st.chunk(2, dim=1)
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        s_mod = s / (s_fac*self.norm_coeff)
        s_mag = torch.abs(s_mod)
        s_phase = torch.angle(s_mod)

        s = (torch.tanh(s_mag)*(torch.cos(s_phase)+1j*torch.sin(s_phase))) * s_fac
        if self.even:
            s = s*(1-mask)
            t = t*(1-mask)
        else:
            s = s*mask
            t = t*mask

        if not sample:
            z = x_f * (torch.exp(s)) + t
            ldj = s.real.sum(dim=[1, 2, 3])
        else:
            z = (x_f - t) / (torch.exp(s))
            ldj = - s.real.sum(dim=[1, 2, 3])

        z = torch.fft.ifft2(z, dim=(-2, -1), s=(h, w), norm='ortho').real
        return z, ldj

    def partition(self, x, mask):
        h, w = x.shape[-2:]
        x_f = torch.fft.fft2(x, dim=(-2, -1), s=(h, w), norm='ortho')
        x_left = x_f.clone()
        x_right = x_f.clone()
        x_left = x_left * mask
        x_right = x_right * (1 - mask)
        x_left = torch.fft.ifft2(x_left, dim=(-2, -1), s=(h, w), norm='ortho').real
        x_right = torch.fft.ifft2(x_right, dim=(-2, -1), s=(h, w), norm='ortho').real
        return x_left, x_right, x_f
    
class CouplingLayerAlias(nn.Module):
    def __init__(self, even, device, free_net, c_in):
        super().__init__()
        self.free_net = free_net
        self.scaling_factor = nn.Parameter(torch.zeros(c_in))
        self.epsilon = 1e-5
        self.even = even
        self.device = device
    def forward(self, x, sample=False, condition=None):
        h, w = x.shape[-2:]
        x_l, x_r, x_real, x_imag = self.real_split(x)
        if self.even:
            x_left = x_l
        else:
            x_left = x_r
        if condition is not None:
            st = self.free_net(torch.cat([x_left, condition], dim=1))
            s, t = st.chunk(2, dim=1)
        else:
            st = self.free_net(x_left)
            s, t = st.chunk(2, dim=1)
        s = torch.fft.rfft2(s, s=(h, w), norm='ortho')
        t = torch.fft.rfft2(t, s=(h, w), norm='ortho')
        if self.even:
            s = s.imag
            t = t.imag
        else:
            s = s.real
            t = t.real
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        s = torch.tanh(s / s_fac) * s_fac
        if not sample:
            if self.even:
                z = torch.complex(x_real, x_imag * torch.exp(s) + t)
            else:
                z = torch.complex(x_real * torch.exp(s) + t, x_imag)
            ldj = s.sum(dim=[1, 2, 3])
        if sample:
            if self.even:
                z = torch.complex(x_real, (x_imag - t) * torch.exp(-s))
            else:
                z = torch.complex((x_real - t) * torch.exp(-s), x_imag)
            ldj = None
        z = torch.fft.irfft2(z, s=(h, w), norm='ortho')
        return z, ldj  
    def real_split(self, x):
        h, w = x.shape[-2:]
        x_real = torch.fft.rfft2(x, s=(h, w), norm='ortho').real
        x_imag = torch.fft.rfft2(x, s=(h, w), norm='ortho').imag
        x_r = torch.fft.irfft2(torch.complex(x_real, torch.zeros_like(x_real)), s=(h, w), norm='ortho').to(x.device)
        x_i = torch.fft.irfft2(torch.complex(torch.zeros_like(x_imag), x_imag), s=(h, w), norm='ortho').to(x.device)
        return x_r, x_i, x_real, x_imag
    
class CouplingPixelLayer(nn.Module):
    def __init__(self, ndv, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encode = nn.Sequential(
            nn.Linear(ndv//2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, ndv),
        )
        self.decode = nn.Sequential(
            nn.Linear(ndv//2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, ndv),
        )
        self.sfac_enc = nn.Parameter(torch.zeros(ndv//2))
        self.sfac_dec = nn.Parameter(torch.zeros(ndv//2))
    def forward(self, x, sample=False):
        x0, x1 = x.chunk(2, dim=-1)
        if not sample:
            s_fac_enc = self.sfac_enc.exp().view(1, 1, -1)
            s_fac_dec = self.sfac_dec.exp().view(1, 1, -1)
            st = self.encode(x0)
            s, t = st.chunk(2, dim=-1)
            s = torch.tanh(s/s_fac_enc) * s_fac_enc
            x1 = torch.exp(s) * x1 + t
            ldj = s.sum(dim=(0, 2))
            st = self.decode(x1)
            s, t = st.chunk(2, dim=-1)
            s = torch.tanh(s/s_fac_dec) * s_fac_dec
            x0 = x0 * torch.exp(s) + t
            ldj += s.sum(dim=(0, 2))
        else:
            s_fac_dec = self.sfac_dec.exp().view(1, 1, -1)
            s_fac_enc = self.sfac_enc.exp().view(1, 1, -1)
            st = self.decode(x1)
            s, t = st.chunk(2, dim=-1)
            s = torch.tanh(s/s_fac_dec) * s_fac_dec
            x0 = torch.exp(-s) * (x0 - t)
            ldj = - s.sum(dim=(0, 2))
            st = self.encode(x0)
            s, t = st.chunk(2, dim=-1)
            s = torch.tanh(s/s_fac_enc) * s_fac_enc
            x1 = torch.exp(-s) * (x1 - t)
            ldj -= s.sum(dim=(0, 2))
        return torch.cat([x0, x1], dim=-1), ldj