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
def swish(x):
    """Swish activation function"""
    return torch.mul(x, torch.sigmoid(x))
class TransformerEncoderLayerPrenorm(TransformerEncoderLayer):
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayerPrenorm, self).__setstate__(state)


    def forward(self, src, src_mask = None, src_key_padding_mask= None, **kwargs):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.norm1(src2)
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.norm2(src2)
        src = src + self.dropout2(src2)
        return src

class TransformerDecoderLayerPrenorm(TransformerDecoderLayer):
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayerPrenorm, self).__setstate__(state)

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
        tgt2 = self.norm2(tgt2)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt2 = self.norm3(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TransformerPrenorm(Transformer):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder = None, custom_decoder = None) -> None:
        super(TransformerPrenorm, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayerPrenorm(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayerPrenorm(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

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
        self.pred_layer = nn.Linear(nde, (ndv+nde)//3)
        self.pred_layer_2 = nn.Linear((ndv+nde)//3, 2*(ndv+nde)//3)
        self.pred_layer_3 = nn.Linear(2*(ndv+nde)//3, ndv)
    def forward(self, x):
        return self.pred_layer_3(swish(self.pred_layer_2(swish(self.pred_layer(x)))))


class PositionEncoder(nn.Module):
    def __init__(self, ndp, nde, mode, init_factor):
        super(PositionEncoder, self).__init__()
        self.mode = mode

        if self.mode == 'linear':
            self.pred_layer = nn.Linear(ndp, nde)
        elif self.mode == 'fourier':
            assert nde%2==0, 'require even emdedding dimension'
            # Based on https://colab.research.google.com/github/tancik/fourier-feature-networks/blob/master/Demo.ipynb
            # self.proj_layer = nn.Parameter(torch.randn(ndp, nde//2)*10)
            self.proj_layer = nn.Parameter(torch.randn(ndp, nde//2)*init_factor)
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