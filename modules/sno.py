import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.constrained import LinearConstrained

class CSplitSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return F.softplus(input.real) + 1j * F.softplus(input.imag)
    
class SNO(nn.Module):
    def __init__(self, in_modes, hid_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = CSplitSoftplus()
    def forward(self, x):
        x_f = torch.fft.rfft2(x, dim=(-2, -1))
        shape = x_f.shape
        x_f = x_f.view(shape[0], -1)

class NN_c(nn.Module):
    def __init__(self, in_dim, n_hid, out_dim, num_hid_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = CSplitSoftplus()
        self.fc = nn.ModuleList([LinearConstrained(in_dim, n_hid, coeff=1.0, iterative_step_manager=None, rayleigh_init=True)]
                                +[LinearConstrained(n_hid, n_hid, coeff=1.0, iterative_step_manager=None, rayleigh_init=True) for _ in range(num_hid_layers)]
                                +[LinearConstrained(n_hid, out_dim, coeff=1.0, iterative_step_manager=None, rayleigh_init=True)])
    def forward(self, x):
        for i in self.fc[:-1]:
            x = self.act(i(x))
        return self.act(self.fc[-1](x))
    
class NN_int(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        