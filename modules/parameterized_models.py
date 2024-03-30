import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

def calc_pointwise(f, flattened_params, n_hid, n_layers, n_in, n_out, nonlinearity):
    index = 0
    shape = [(n_in, n_hid)] + [(n_hid, n_hid)]*(n_layers-2) + [(n_hid, n_out)]
    b_size = flattened_params.shape[0]
    for i in shape:
        param_size = np.prod(i)+i[-1]
        weight = flattened_params[:, index:index+np.prod(i)].reshape(b_size, *i)
        bias = flattened_params[:, index+np.prod(i):index+param_size].reshape(b_size, 1, i[-1])
        if i != len(shape) - 1:
            f = torch.matmul(f, weight)
            f = f + bias
            f = nonlinearity(f)
        else:
            f = torch.matmul(f, weight)
            f = f + bias
        index += param_size
    return f

class PointwiseLoad(nn.Module):
    def __init__(self, n_hid, n_layers, n_in, n_out, nonlinearity, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.n_in = n_in
        self.n_out = n_out
        self.nonlinearity = nonlinearity
    def forward(self, x, flattened_params):
        return calc_pointwise(x, flattened_params, self.n_hid, self.n_layers, self.n_in, self.n_out, self.nonlinearity)