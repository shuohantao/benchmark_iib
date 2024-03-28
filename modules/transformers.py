from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pdb
import math
import torch
import torch.nn as nn

from torch.nn.modules.transformer import *
import torch.nn as nn
import torch.nn.functional as F
LayerNorm = nn.LayerNorm
        
class TransformerDecoderLayerNoSelfAttn(TransformerDecoderLayer):
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayerNoSelfAttn, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None,
                tgt_key_padding_mask = None, memory_key_padding_mask = None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    
class TransformerEfficient(Transformer):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder = None, custom_decoder = None) -> None:
        super(TransformerEfficient, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayerNoSelfAttn(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead



class ValueEncoder(nn.Module):
    def __init__(self, ndv, nde):
        super(ValueEncoder, self).__init__()
        self.pred_layer = nn.Linear(ndv, nde)

    def forward(self, x):
        return self.pred_layer(x)


class ValueDecoder(nn.Module):
    def __init__(self, nde, ndv):
        super(ValueDecoder, self).__init__()
        self.pred_layer = nn.Linear(nde, ndv)

    def forward(self, x):
        return self.pred_layer(x)


class PositionEncoder(nn.Module):
    def __init__(self, ndp, nde, posnenc_cfg):
        super(PositionEncoder, self).__init__()
        self.mode = posnenc_cfg.mode

        if self.mode == 'linear':
            self.pred_layer = nn.Linear(ndp, nde)
        elif self.mode == 'fourier':
            assert nde%2==0, 'require even emdedding dimension'
            # Based on https://colab.research.google.com/github/tancik/fourier-feature-networks/blob/master/Demo.ipynb
            # self.proj_layer = nn.Parameter(torch.randn(ndp, nde//2)*10)
            self.proj_layer = nn.Parameter(torch.randn(ndp, nde//2)*posnenc_cfg.init_factor)
            self.proj_layer.requires_grad = False
    
    def forward(self, x):
        '''
        Args:
            x: ... X ndp
        Returns:
            enc: ... X nde
        '''
        mode = self.mode
        if mode == 'linear':
            return self.pred_layer(x)
        elif mode == 'fourier':
            x_proj = torch.matmul(2*math.pi*x, self.proj_layer)
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)


def square_id_mask(n, device='cpu'):
    mask = torch.eye(n)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.to(device)
    return mask


class S2SNet(nn.Module):
    def __init__(
        self, ndp, ndv, nde, transformer_cfg, posnenc_cfg,
        ndv_out = None, out_pde_fn=distribution_util.Identity
    ):
        super(S2SNet, self).__init__()
        self.ndp = ndp
        self.ndv = ndv
        self.nde = nde
        if ndv_out is None:
            ndv_out = self.ndv

        self.ndv_out = ndv_out
        self.out_pde_fn = out_pde_fn

        self.val_enc = ValueEncoder(ndv, nde)
        self.val_dec = ValueDecoder(2*nde, ndv_out)
        self.pos_enc = PositionEncoder(ndp, nde, posnenc_cfg)
        Transformer = TransformerEfficient
        self.transformer = Transformer(
            nhead=transformer_cfg.nhead,
            num_encoder_layers=transformer_cfg.num_encoder_layers,
            num_decoder_layers=transformer_cfg.num_decoder_layers,
            dim_feedforward=4*nde,
            d_model = 2*nde
        )

    def forward(self, sample_vals, sample_posns, query_posns):
        '''
        Args:
            sample_vals: nS X B X ndv
            sample_posns: nS X B X ndp
            query_posns: nQ X B X ndp
        Returns:
            vals: prob distribution over (nQ X B X ndv)
        '''
        sample_posns = self.pos_enc(sample_posns)
        sample_vals = self.val_enc(sample_vals)
        query_posns = self.pos_enc(query_posns)

        samples = torch.cat([sample_vals, sample_posns], dim=-1)
        query = torch.cat([torch.zeros_like(query_posns), query_posns], dim=-1)
        tgt_mask = square_id_mask(query.shape[0], device=query.device)

        query_vals = self.transformer(samples, query, tgt_mask=tgt_mask)
        return self.out_pde_fn(self.val_dec(query_vals))

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