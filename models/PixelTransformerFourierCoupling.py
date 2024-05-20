import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from modules.transformers import *
from modules.coupling import CouplingPixelLayer
from modules.act_norm import ActNormSeq
from modules.conv import InvConv1d
from math import log2, exp

class PixelTransformerFourierCoupling(nn.Module):
    def __init__(
        self, ndp, ndv, nde, nhead, num_flow_layers, num_encoder_layers, num_decoder_layers, mode, init_factor, context_num=4, sample_mean=False):
        super(PixelTransformerFourierCoupling, self).__init__()
        self.ndp = ndp
        self.ndv = ndv
        self.nde = nde
        self.sample_mean = sample_mean
        self.num_layer = 32
        self.ndv_out = 4
        self.val_enc = ValueEncoder(ndv, nde)
        self.val_dec = ValueDecoder(2*nde, self.ndv_out)
        self.pos_enc = PositionEncoder(ndp, nde, mode, init_factor)
        Transformer = TransformerPrenorm
        self.transformer = Transformer(
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=4*nde,
            d_model = 2*nde
        )
        
        self.context_num = context_num
        self.flows = nn.ModuleList([i for _ in range(num_flow_layers) for i in [ActNormSeq(2), InvConv1d(2), CouplingPixelLayer(ndv)]])
        self.prior = torch.distributions.Normal(0, 1)
        self.base_res = context_num
        self.ar_step = 2
        self.query_token = nn.Parameter(torch.randn(1, 1, nde))
        self.lmd = nn.Parameter(torch.ones(1))
    def calc_flow(self, x, truth, scale=1):
        sig = x[..., :2].exp()
        mu = x[..., -2:]
        z = self.prior.sample(sig.shape).to(x.device)
        z = z * sig + mu
        for flow in reversed(self.flows):
            z, _ = flow(z, sample=True)
        z = (F.tanh(z) * 255 * 1.5 - 255 * 0.5)*scale
        scaled_residual = (z - truth)
        return torch.norm(scaled_residual), z
    def sample_flow(self, x, scale):
        sig = x[..., :2].exp()
        mu = x[..., -2:]
        z = self.prior.sample(sig.shape).to(x.device)
        z = z * sig + mu
        for flow in reversed(self.flows):
            z, _ = flow(z, sample=True)
        z = (F.tanh(z) * 255 * 1.5 - 255 * 0.5)*scale
        return z
    # def calc_flow_bpd(self, x):
    #     sig = x[..., :2].exp()
    #     mu = x[..., -2:]
    #     z = self.prior.sample(sig.shape).to(x.device)
    #     sldj = self.prior.log_prob(z).sum(dim=(0, 2))
    #     l0 = sldj.sum()
    #     z = z * sig + mu
    #     sldj -= x[..., :2].sum(dim=(0, 2))
    #     l1 = x[..., :2].sum()
    #     for flow in reversed(self.flows):
    #         z, ldj = flow(z, sample=True)
    #         sldj += ldj
    #     z = torch.atanh(z)
    #     sldj -= (1-z**2).log().sum(dim=(0, 2))
    #     l3 = (1-z**2).log().sum()
    #     return sldj.sum()*log2(exp(1)) / (z.shape[0] * z.shape[1] * z.shape[2]), [l0, l1, l3], [torch.max(sig), torch.min(sig), torch.max(mu), torch.min(mu)]
    def sample_context(self, x, pos, scale=None, num=4):
        b, c, h, w = x.shape
        indices = torch.arange(h*w).view(h, w)
        indices = torch.cat((indices[:num//2, :num//2].flatten(), indices[-num//2:, :num//2].flatten()))
        x = x.flatten(start_dim=-2, end_dim=-1).permute(2, 0, 1)
        pos = pos.flatten(start_dim=-2, end_dim=-1).permute(2, 0, 1)
        
        vals = x[indices]
        posns = pos[indices]
        q_mask = torch.ones(h*w, dtype=torch.bool)
        q_mask[indices] = False
        remaining_vals = x[q_mask]
        remaining_posns = pos[q_mask]
        if scale is not None:
            scale = scale.flatten(start_dim=-2, end_dim=-1).permute(2, 0, 1)
            remaining_scales = scale[q_mask]
        else:
            remaining_scales = None
        return vals, posns, remaining_vals, remaining_posns, indices, remaining_scales
    def sample_ar_context(self, x, pos, step, use_lower=False):
        b, c, h, w = x.shape
        indices = torch.arange(h*w).view(h, w)
        indices = torch.cat((indices[:(h-step)//2, :(h-step)//2].flatten(), indices[-(h-step)//2:, :(h-step)//2].flatten()))
        if use_lower:
            ctx_mask = indices
        else:
            ctx_idx = torch.arange(h*w).view(h, w)
            ctx_idx = torch.cat((indices[:(h-2*step)//2, :(h-2*step)//2].flatten(), indices[-(h-2*step)//2:, :(h-2*step)//2].flatten()))
            ctx_mask = torch.zeros(h*w, dtype=torch.bool)
            ctx_mask[indices] = True
            ctx_mask[ctx_idx] = False
        x = x.flatten(start_dim=-2, end_dim=-1).permute(2, 0, 1)
        pos = pos.flatten(start_dim=-2, end_dim=-1).permute(2, 0, 1)
        vals = x[ctx_mask]
        posns = pos[ctx_mask]
        q_mask = torch.ones(h*w, dtype=torch.bool)
        q_mask[indices] = False
        remaining_vals = x[q_mask]
        remaining_posns = pos[q_mask]
        return vals, posns, remaining_vals, remaining_posns, indices
    
    def forward(self, x):
        '''
        Args:
            sample_vals: nS X B X ndv
            sample_posns: nS X B X ndp
            query_posns: nQ X B X ndp
        Returns:
            vals: prob distribution over (nQ X B X ndv)
        '''
        x = torch.fft.rfft2(x, dim=(-2, -1), norm='forward')
        x = torch.cat([x.real, x.imag], dim=1)
        batch_size, _, height, width = x.shape
        x_coords = torch.arange(width, dtype=torch.float32, device=x.device).view(1, 1, 1, -1)
        y_coords = torch.arange(height, dtype=torch.float32, device=x.device).view(1, 1, -1, 1)
        pos = torch.zeros((batch_size, 2, height, width), dtype=torch.float32, device=x.device)
        pos[:, 0, :, :] = x_coords/self.base_res
        pos[:, 1, :, :] = y_coords/self.base_res
        pos2d = pos.clone()
        height = pos2d.shape[-2]
        midpoint = height // 2
        pos2d[:, 1, midpoint:, :] = torch.flip(pos2d[:, 1, :midpoint, :], [-2])
        scale = torch.maximum(pos2d[:, 0, ...], pos2d[:, 1, ...])
        scale = torch.exp(- self.lmd*scale).unsqueeze(1)
        sample_vals, sample_posns, query_truth, query_posns, _, query_scales = self.sample_context(x, pos, scale=scale, num=self.context_num)
        sample_posns = self.pos_enc(sample_posns)
        sample_vals = self.val_enc(sample_vals)
        query_posns = self.pos_enc(query_posns)
        samples = torch.cat([sample_vals, sample_posns], dim=-1)
        query = torch.cat([self.query_token.expand(query_posns.shape[0], query_posns.shape[1], -1), query_posns], dim=-1)
        query_vals = self.transformer(samples, query)
        params = self.val_dec(query_vals)
        nll = self.calc_flow(params, query_truth, query_scales)[0]
        return nll.sum()
    
    # def forward(self, x):
    #     res = x.shape[-1]
    #     x = torch.fft.rfft2(x, dim=(-2, -1), norm='forward')
    #     x = torch.cat([x.real, x.imag], dim=1)
    #     batch_size, _, height, width = x.shape
    #     x_coords = torch.arange(width, dtype=torch.float32, device=x.device).view(1, 1, 1, -1)
    #     y_coords = torch.arange(height, dtype=torch.float32, device=x.device).view(1, 1, -1, 1)
    #     pos = torch.zeros((batch_size, 2, height, width), dtype=torch.float32, device=x.device)
    #     pos[:, 0, :, :] = x_coords / self.base_res
    #     pos[:, 1, :, :] = y_coords / self.base_res
    #     f_low = self.base_res
    #     total_loss = 0
    #     raw_vals, raw_posns, _ = self.slice_freq(x, pos, 0, f_low)
    #     for i in range((res-self.base_res)//2):
    #         truth_val, query_posns_raw, _ = self.slice_freq(x, pos, f_low, f_low+2)
    #         sample_posns = self.pos_enc(raw_posns)
    #         sample_vals = self.val_enc(raw_vals)
    #         query_posns = self.pos_enc(query_posns_raw)
    #         samples = torch.cat([sample_vals, sample_posns], dim=-1)
    #         query = torch.cat([torch.zeros_like(query_posns), query_posns], dim=-1)
    #         query_vals = self.transformer(samples, query)
    #         params = self.val_dec(query_vals)
    #         nll, new_vals = self.calc_flow(params, truth_val)
    #         raw_posns = torch.cat([raw_posns, query_posns_raw], dim=0)
    #         raw_vals = torch.cat([raw_vals, new_vals], dim=0)
    #         total_loss += nll
    #     return total_loss.sum()
    
    def sample(self, x, num_context, autoregessive=False, num_samples=-1, *args, **kwargs):
        if not autoregessive:
            return self.sample_normal(x, num_context, num_samples=num_samples, *args, **kwargs)
        else:
            return self.sample_ar(x, num_context, *args, **kwargs)
        
    def sample_normal(self, x, num_context, num_samples, *args, **kwargs):
        x = x[:num_samples, ...]
        x = torch.fft.rfft2(x, dim=(-2, -1), norm='forward')
        x = torch.cat([x.real, x.imag], dim=1)
        batch_size, channel, height, width = x.shape
        x_coords = torch.arange(width, dtype=torch.float32, device=x.device).view(1, 1, 1, -1)
        y_coords = torch.arange(height, dtype=torch.float32, device=x.device).view(1, 1, -1, 1)
        pos = torch.zeros((batch_size, 2, height, width), dtype=torch.float32, device=x.device)
        pos[:, 0, :, :] = x_coords / self.base_res
        pos[:, 1, :, :] = y_coords / self.base_res
        pos2d = pos.clone()
        height = pos2d.shape[-2]
        midpoint = height // 2
        pos2d[:, 1, midpoint:, :] = torch.flip(pos2d[:, 1, :midpoint, :], [-2])
        scale = torch.maximum(pos2d[:, 0, ...], pos2d[:, 1, ...])
        scale = torch.exp(- self.lmd*scale).unsqueeze(1)
        sample_truth, sample_posns, query_truth, query_posns, idx, scale = self.sample_context(x, pos, num=num_context, scale=scale)
        x = x.flatten(start_dim=-2, end_dim=-1).permute(2, 0, 1)
        pos = pos.flatten(start_dim=-2, end_dim=-1).permute(2, 0, 1)
        sample_posns = self.pos_enc(sample_posns)
        sample_vals = self.val_enc(sample_truth)
        query_posns = self.pos_enc(query_posns)
        samples = torch.cat([sample_vals, sample_posns], dim=-1)
        query = torch.cat([self.query_token.expand(query_posns.shape[0], query_posns.shape[1], -1), query_posns], dim=-1)
        query_vals = self.transformer(samples, query)
        pS, pB, _ = query_vals.shape
        query_vals = self.val_dec(query_vals)
        query_vals = self.sample_flow(query_vals, scale)
        img = x.clone()
        fill_idx = torch.ones(height*width, dtype=torch.bool)
        fill_idx[idx] = False
        img[fill_idx] = query_vals

        img = torch.complex(img[..., 0], img[..., 1]).unsqueeze(-1)
        img = img.permute(1, 2, 0).view(batch_size, channel//2, height, width)
        img = torch.fft.irfft2(img, dim=(-2, -1), norm='forward')
        return img

    def sample_ar(self, x, *args, **kwargs):
        res = x.shape[-1]
        x = torch.fft.rfft2(x, dim=(-2, -1), norm='forward')
        x = torch.cat([x.real, x.imag], dim=1)
        batch_size, _, height, width = x.shape
        x_coords = torch.arange(width, dtype=torch.float32, device=x.device).view(1, 1, 1, -1)
        y_coords = torch.arange(height, dtype=torch.float32, device=x.device).view(1, 1, -1, 1)
        pos = torch.zeros((batch_size, 2, height, width), dtype=torch.float32, device=x.device)
        pos[:, 0, :, :] = x_coords / self.base_res
        pos[:, 1, :, :] = y_coords / self.base_res
        img = torch.zeros_like(x.flatten(start_dim=-2, end_dim=-1).permute(2, 0, 1))
        f_low = self.base_res
        sample_vals, sample_posns, _ = self.slice_freq(x, pos, 0, f_low)
        for i in range((res-self.base_res)//2):
            truth_val, query_posns, q_mask = self.slice_freq(x, pos, f_low, f_low+2)
            sample_posns = self.pos_enc(sample_posns)
            sample_vals = self.val_enc(sample_vals)
            query_posns = self.pos_enc(query_posns)
            samples = torch.cat([sample_vals, sample_posns], dim=-1)
            query = torch.cat([self.query_token.expand(query_posns.shape[0], query_posns.shape[1], -1), query_posns], dim=-1)
            query_vals = self.transformer(samples, query)
            params = self.val_dec(query_vals)
            new_vals = self.sample_flow(params)
            img[q_mask] = new_vals
            sample_posns = torch.cat([sample_posns, query_posns], dim=0)
            sample_vals = torch.cat([sample_vals, new_vals], dim=0)
        img = torch.complex(img[..., 0], img[..., 1]).unsqueeze(-1)
        img = img.permute(1, 2, 0).view(x.shape[0], x.shape[1]//2, res, res)
        img = torch.fft.irfft2(img, dim=(-2, -1), norm='forward')
        return img
    
    def slice_freq(self, x, pos, f1, f2):
        b, c, h, w = x.shape
        indices = torch.arange(h*w).view(h, w)
        indices = torch.cat((indices[f1//2:f2//2, :f2//2].flatten(), indices[:f1//2, f1//2:f2//2].flatten(), indices[-f2//2:, f1//2:f2//2].flatten(), indices[f1//2:f2//2, -f1//2].flatten()))

        x = x.flatten(start_dim=-2, end_dim=-1).permute(2, 0, 1)
        pos = pos.flatten(start_dim=-2, end_dim=-1).permute(2, 0, 1)
        q_mask = torch.zeros(h*w, dtype=torch.bool)
        q_mask[indices] = True
        vals = x[q_mask]
        posns = pos[q_mask]
        return vals, posns, q_mask