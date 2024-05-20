import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from modules.transformers import *
from modules.coupling import CouplingPixelLayer
from modules.act_norm import ActNormSeq
from modules.resnet import ResNetMLP
class PixelFourierVAE(nn.Module):
    def __init__(
        self, ndp, ndv, nde, nhead, num_encoder_layers, num_decoder_layers, mode, init_factor, context_num=4, sample_mean=False):
        super(PixelFourierVAE, self).__init__()
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
        self.vae_decoder = ResNetMLP(nde, 2*ndv, self.num_layer)
    def calc_vae(self, x, truth):
        sig = x[..., :2]
        mu = x[..., -2:]
        # kl = -0.5 * torch.sum(1 + sig - mu.pow(2) - sig.exp())
        x = torch.randn_like(sig) * sig.exp() + mu
        x = self.vae_decoder(x)
        x = F.tanh(x) * 255 * 1.5 - 255 * 0.5
        bce = F.mse_loss(x, truth, reduction='sum')
        return bce
    def sample_vae(self, x):
        sig = x[..., :2]
        mu = x[..., -2:]
        x = torch.randn_like(sig) * sig.exp() + mu
        x = self.vae_decoder(x)
        x = F.tanh(x) * 255 * 1.5 - 255 * 0.5
        return x
    def sample_context(self, x, pos, num=4):
        nS, B, ndv = x.shape
        if num == -1:
            num = torch.randint(1, nS//4)
        indices = torch.randperm(nS)[:num]
        vals = x[indices]
        posns = pos[indices]
        remaining_vals = torch.cat([x.unsqueeze(0) for i, x in enumerate(x) if i not in indices], dim=0)
        remaining_posns = torch.cat([x.unsqueeze(0) for i, x in enumerate(pos) if i not in indices], dim=0)
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
        x = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        x = torch.cat([x.real, x.imag], dim=1)
        batch_size, _, height, width = x.shape
        x_coords = torch.arange(width, dtype=torch.float32, device=x.device).view(1, 1, 1, -1)
        y_coords = torch.arange(height, dtype=torch.float32, device=x.device).view(1, 1, -1, 1)
        pos = torch.zeros((batch_size, 2, height, width), dtype=torch.float32, device=x.device)
        pos[:, 0, :, :] = x_coords / width
        pos[:, 1, :, :] = y_coords / height
        x = x.flatten(start_dim=-2, end_dim=-1).permute(2, 0, 1)
        pos = pos.flatten(start_dim=-2, end_dim=-1).permute(2, 0, 1)
        sample_vals, sample_posns, query_truth, query_posns, _ = self.sample_context(x, pos, num=self.context_num)
        sample_posns = self.pos_enc(sample_posns)
        sample_vals = self.val_enc(sample_vals)
        query_posns = self.pos_enc(query_posns)
        samples = torch.cat([sample_vals, sample_posns], dim=-1)
        query = torch.cat([torch.zeros_like(query_posns), query_posns], dim=-1)
        query_vals = self.transformer(samples, query)
        params = self.val_dec(query_vals)
        nll = self.calc_vae(params, query_truth)
        return nll.sum()
    def sample(self, x, num_context, autoregessive=False, *args, **kwargs):
        if not autoregessive:
            return self.sample_normal(x, num_context, *args, **kwargs)
        else:
            return self.sample_conditional(x, num_context, *args, **kwargs)
    def sample_normal(self, x, num_context, *args, **kwargs):
        orig_shape = x.shape[-2:]
        x = torch.fft.rfft2(x, dim=(-2, -1), s=orig_shape, norm='ortho')
        x = torch.cat([x.real, x.imag], dim=1)
        batch_size, channel, height, width = x.shape
        x_coords = torch.arange(width, dtype=torch.float32, device=x.device).view(1, 1, 1, -1)
        y_coords = torch.arange(height, dtype=torch.float32, device=x.device).view(1, 1, -1, 1)
        pos = torch.zeros((batch_size, 2, height, width), dtype=torch.float32, device=x.device)
        pos[:, 0, :, :] = x_coords / width
        pos[:, 1, :, :] = y_coords / height
        x = x.flatten(start_dim=-2, end_dim=-1).permute(2, 0, 1)
        pos = pos.flatten(start_dim=-2, end_dim=-1).permute(2, 0, 1)
        sample_truth, sample_posns, query_truth, query_posns, idx = self.sample_context(x, pos, num=num_context)
        sample_posns = self.pos_enc(sample_posns)
        sample_vals = self.val_enc(sample_truth)
        query_posns = self.pos_enc(query_posns)
        samples = torch.cat([sample_vals, sample_posns], dim=-1)
        query = torch.cat([torch.zeros_like(query_posns), query_posns], dim=-1)
        query_vals = self.transformer(samples, query)
        pS, pB, _ = query_vals.shape
        query_vals = self.val_dec(query_vals)
        query_vals = self.sample_vae(query_vals)
        img = torch.zeros_like(x).to(x.device)
        count = 0
        for i in range(height*width):
            if i in idx:
                # img[i, ...] = x[i, ...]
                img[i, ...] = torch.ones_like(x[i, ...])
            else:
                img[i, ...] = query_vals[count, ...]
                count += 1

        img = torch.complex(img[..., 0], img[..., 1]).unsqueeze(-1)
        img = img.permute(1, 2, 0).view(batch_size, channel//2, height, width)
        img = torch.fft.irfft2(img, dim=(-2, -1), s=orig_shape, norm='ortho')
        return img
    def sample_conditional(self, x, num_context, resolution, *args, **kwargs):
        '''
        Args:
            sample_vals: nS X B X ndv
            sample_posns: nS X B X ndp
            query_posns: nQ X B X ndp
        Returns:
            vals: nQ X B X ndv
        Infers vals one query at a time, adding back the prediction to samples
        '''
        orig_shape = x.shape[-2:]
        x = torch.fft.rfft2(x, dim=(-2, -1), s=orig_shape, norm='ortho')
        x = torch.cat([x.real, x.imag], dim=1)
        query_vals = []
        batch_size, channel, height, width = x.shape
        with torch.no_grad():
            x_coords = torch.arange(width, dtype=torch.float32, device=x.device).view(1, 1, 1, -1)
            y_coords = torch.arange(height, dtype=torch.float32, device=x.device).view(1, 1, -1, 1)
            pos = torch.zeros((batch_size, 2, height, width), dtype=torch.float32, device=x.device)
            pos[:, 0, :, :] = x_coords / width
            pos[:, 1, :, :] = y_coords / height
            x = x.flatten(start_dim=-2, end_dim=-1).permute(2, 0, 1)
            pos = pos.flatten(start_dim=-2, end_dim=-1).permute(2, 0, 1)
            sample_vals, sample_posns, query_truth, query_posns, idx = self.sample_ordered_context(x, pos, num=num_context)
            sample_posns = self.pos_enc(sample_posns)
            sample_vals = self.val_enc(sample_vals)
            query_posns = self.pos_enc(query_posns)
            ndq = query_posns.shape[0]
            samples = torch.cat([sample_vals, sample_posns], dim=-1)
            zero_vec = torch.zeros_like(query_posns[[0]])

            for qx in range(ndq):
                query = torch.cat([zero_vec, query_posns[[qx]]], dim=-1)
                # tgt_mask = square_id_mask(1, device=query.device)

                query_val = self.transformer(samples, query)
                query_val = self.val_dec(query_val)
                query_val = self.sample_vae(query_val)
                query_val_enc = self.val_enc(query_val)
                sample_new = torch.cat([query_val_enc, query_posns[[qx]]], dim=-1)
                samples = torch.cat([samples, sample_new], dim=0)
                samples = samples[-self.num_context:]
                query_vals.append(query_val)
            query_vals = torch.cat(query_vals, dim=0)
            img = torch.zeros_like(x).to(x.device)
            count = 0
            for i in range(height*width):
                if i in idx:
                    # img[i, ...] = x[i, ...]
                    img[i, ...] = torch.ones_like(x[i, ...])
                else:
                    img[i, ...] = query_vals[count, ...]
                    count += 1
            img = torch.complex(img[..., 0], img[..., 1]).unsqueeze(-1)
            img = img.permute(1, 2, 0).view(batch_size, channel//2, height, width)
            img = torch.fft.irfft2(img, dim=(-2, -1), s=orig_shape, norm='ortho')
            return img
    def sample_ordered_context(self, x, pos, num=4):
        nS, B, ndv = x.shape
        if num == -1:
            num = torch.randint(1, nS//4)
        indices = torch.randperm(nS)[:num]
        indices = torch.arange(0, num)
        vals = x[indices]
        posns = pos[indices]
        remaining_vals = torch.cat([x.unsqueeze(0) for i, x in enumerate(x) if i not in indices], dim=0)
        remaining_posns = torch.cat([x.unsqueeze(0) for i, x in enumerate(pos) if i not in indices], dim=0)
        return vals, posns, remaining_vals, remaining_posns, indices