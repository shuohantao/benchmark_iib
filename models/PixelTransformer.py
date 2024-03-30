import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from modules.transformers import *
from modules.mog import MoG, MoGNearest



class PixelTransformer(nn.Module):
    def __init__(
        self, ndp, ndv, nde, nhead, num_encoder_layers, num_decoder_layers, mode, init_factor, num_bins, context_num=4):
        super(PixelTransformer, self).__init__()
        self.ndp = ndp
        self.ndv = ndv
        self.nde = nde

        self.ndv_out = num_bins*3

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
        self.bins = torch.linspace(0, 255, num_bins).cuda()
        self.mog = MoG(self.bins)
        self.context_num = context_num
    def sample_random_context(self, x, pos, num=4):
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
        batch_size, _, height, width = x.shape
        x_coords = torch.arange(width, dtype=torch.float32, device=x.device).view(1, 1, 1, -1)
        y_coords = torch.arange(height, dtype=torch.float32, device=x.device).view(1, 1, -1, 1)
        pos = torch.zeros((batch_size, 2, height, width), dtype=torch.float32, device=x.device)
        pos[:, 0, :, :] = x_coords / width
        pos[:, 1, :, :] = y_coords / height
        x = x.flatten(start_dim=-2, end_dim=-1).permute(2, 0, 1)
        pos = pos.flatten(start_dim=-2, end_dim=-1).permute(2, 0, 1)
        sample_vals, sample_posns, query_truth, query_posns, _ = self.sample_random_context(x, pos, num=self.context_num)
        sample_posns = self.pos_enc(sample_posns)
        sample_vals = self.val_enc(sample_vals)
        query_posns = self.pos_enc(query_posns)
        samples = torch.cat([sample_vals, sample_posns], dim=-1)
        query = torch.cat([torch.zeros_like(query_posns), query_posns], dim=-1)
        tgt_mask = square_id_mask(query.shape[0], device=query.device)
        query_vals = self.transformer(samples, query, tgt_mask=tgt_mask)
        tau, mu, sigs = self.val_dec(query_vals).chunk(3, dim=-1)
        sigs = F.softplus(sigs) + 1e-5
        nll = -self.mog.log_prob(tau, mu, sigs, query_truth)
        return nll.sum()
    def sample(self, x):
        batch_size, channel, height, width = x.shape
        x_coords = torch.arange(width, dtype=torch.float32, device=x.device).view(1, 1, 1, -1)
        y_coords = torch.arange(height, dtype=torch.float32, device=x.device).view(1, 1, -1, 1)
        pos = torch.zeros((batch_size, 2, height, width), dtype=torch.float32, device=x.device)
        pos[:, 0, :, :] = x_coords / width
        pos[:, 1, :, :] = y_coords / height
        x = x.flatten(start_dim=-2, end_dim=-1).permute(2, 0, 1)
        pos = pos.flatten(start_dim=-2, end_dim=-1).permute(2, 0, 1)
        sample_truth, sample_posns, query_truth, query_posns, idx = self.sample_random_context(x, pos, num=self.context_num)
        sample_posns = self.pos_enc(sample_posns)
        sample_vals = self.val_enc(sample_truth)
        query_posns = self.pos_enc(query_posns)
        samples = torch.cat([sample_vals, sample_posns], dim=-1)
        query = torch.cat([torch.zeros_like(query_posns), query_posns], dim=-1)
        tgt_mask = square_id_mask(query.shape[0], device=query.device)
        query_vals = self.transformer(samples, query, tgt_mask=tgt_mask)
        tau, mu, sigs = self.val_dec(query_vals).chunk(3, dim=-1)
        sigs = F.softplus(sigs) + 1e-5
        tau = F.softmax(tau, dim=-1)
        # qS, qB, qV = query_truth.shape
        # query_idx = torch.multinomial(tau.view(-1, tau.shape[-1]), num_samples=1).view(qS, qB, qV)
        # query_vals = self.bins[query_idx]
        # mu_selected = torch.gather(mu, dim=2, index=query_idx.long())
        # sigs_selected = torch.gather(sigs, dim=2, index=query_idx.long())
        # query_vals = query_vals + mu_selected + torch.randn_like(mu_selected) * sigs_selected
        query_vals = (tau * (self.bins + mu)).sum(dim=-1, keepdim=True)
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
    def sample_conditional(self, sample_vals, sample_posns, query_posns, tau=1.0):
        '''
        Args:
            sample_vals: nS X B X ndv
            sample_posns: nS X B X ndp
            query_posns: nQ X B X ndp
        Returns:
            vals: nQ X B X ndv
        Infers vals one query at a time, adding back the prediction to samples
        '''
        ndq = query_posns.shape[0]
        query_vals = []
        with torch.no_grad():
            sample_posns = self.pos_enc(sample_posns)
            sample_vals = self.val_enc(sample_vals)
            query_posns = self.pos_enc(query_posns)
            samples = torch.cat([sample_vals, sample_posns], dim=-1)
            zero_vec = torch.zeros_like(query_posns[[0]])

            for qx in range(ndq):
                query = torch.cat([zero_vec, query_posns[[qx]]], dim=-1)
                tgt_mask = square_id_mask(1, device=query.device)

                query_val = self.transformer(samples, query, tgt_mask=tgt_mask)
                query_val = self.out_pde_fn(self.val_dec(query_val)).sample(tau=tau)

                query_val_enc = self.val_enc(query_val)
                sample_new = torch.cat([query_val_enc, query_posns[[qx]]], dim=-1)

                samples = torch.cat([samples, sample_new], dim=0)
                query_vals.append(query_val)

            return torch.cat(query_vals, dim=0)