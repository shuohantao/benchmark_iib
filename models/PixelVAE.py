import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from modules.transformers import *
from modules.mog import MoG, MoGNearest
from modules.augmented import AugmentedParamFlow

class PixelVAE(nn.Module):
    def __init__(
        self, ndp, ndv, nde, nhead, num_encoder_layers, num_decoder_layers, mode, init_factor, context_num=4, sample_mean=False):
        super(PixelVAE, self).__init__()
        self.ndp = ndp
        self.ndv = ndv
        self.nde = nde
        self.sample_mean = sample_mean
        self.num_layer = 32
        self.ndv_out = 2
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
        self.vae_decoder = nn.Sequential(
            nn.Linear(ndv, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, ndv),
            nn.Sigmoid()
        )
        self.context_num = context_num
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
    def forward(self, x):
        '''
        Args:
            sample_vals: nS X B X ndv
            sample_posns: nS X B X ndp
            query_posns: nQ X B X ndp
        Returns:
            vals: prob distribution over (nQ X B X ndv)
        '''
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
        # tgt_mask = square_id_mask(query.shape[0], device=query.device)
        query_vals = self.transformer(samples, query)
        params = self.val_dec(query_vals)
        nll = self.calc_vae(params, query_truth)
        return nll.sum()
    def calc_vae(self, x, truth):
        sig = x[..., 0]
        mu = x[..., 1]
        # kl = -0.5 * torch.sum(1 + sig - mu.pow(2) - sig.exp())
        x = torch.randn_like(sig) * sig.exp() + mu
        x = self.vae_decoder(x.unsqueeze(-1))
        bce = F.binary_cross_entropy(x, truth/255, reduction='sum')
        return bce
    def sample_vae(self, x):
        sig = x[..., 0]
        mu = x[..., 1]
        x = torch.randn_like(sig) * sig.exp() + mu
        x = self.vae_decoder(x.unsqueeze(-1))*255
        return x
    def sample(self, x, num_context, autoregessive=False, *args, **kwargs):
        if not autoregessive:
            return self.sample_normal(x, num_context, *args, **kwargs)
        else:
            return self.sample_conditional(x, num_context, *args, **kwargs)
    def sample_normal(self, x, num_context, *args, **kwargs):
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
        query_vals = self.val_dec(query_vals)
        query_vals = self.sample_vae(query_vals)
        img = torch.zeros_like(x).to(x.device)
        count = 0
        for i in range(height*width):
            if i in idx:
                img[i, ...] = x[i, ...]
            else:
                img[i, ...] = query_vals[count, ...]
                count += 1
        img = img.permute(1, 2, 0).view(batch_size, channel, height, width)
        return img
    def sample_conditional(self, x, num_context, *args, **kwargs):
        '''
        Args:
            sample_vals: nS X B X ndv
            sample_posns: nS X B X ndp
            query_posns: nQ X B X ndp
        Returns:
            vals: nQ X B X ndv
        Infers vals one query at a time, adding back the prediction to samples
        '''
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
                    img[i, ...] = x[i, ...]
                else:
                    img[i, ...] = query_vals[count, ...]
                    count += 1
            img = img.permute(1, 2, 0).view(batch_size, channel, height, width)
            return img