import torch.nn as nn
import torch
class FreqNorm(nn.Module):
    def __init__(self, n_hid=32, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_hid = n_hid
        self.scale = nn.Sequential(nn.Linear(1, n_hid), nn.ReLU(), nn.Linear(n_hid, 1), nn.ReLU())
        self.bias = nn.Sequential(nn.Linear(1, n_hid), nn.ELU(), nn.Linear(n_hid, 1))
    def forward(self, x, sample=False):
        freq = torch.Tensor([x.shape[-1]]).to(x.device)
        ldj = self.scale(freq).sum() * torch.ones(x.shape[0], device=x.device)
        if not sample:
            x = x * self.scale(freq).exp()+ self.bias(freq)
        else:
            x = (x - self.bias(freq))/self.scale(freq).exp()
            ldj *= -1
        return x, ldj
