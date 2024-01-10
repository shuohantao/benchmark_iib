import torch
from torch.distributions.multivariate_normal import MultivariateNormal

device='cuda' if torch.cuda.is_available() else 'cpu'

class BaseDistManager:
    def __init__(self, type, ls, var, eps):
        self.type = type
        self.base_dist_cache = {}

    def get_base_dist(self, base_shape):
        h_base, w_base = base_shape[-2:]
        if h_base in self.base_dist_cache and w_base in self.base_dist_cache[h_base]:
             return self.base_dist_cache[h_base][w_base]

        if self.type == 'standard':
            base_dist = self._get_base_dist_standard(h_base, w_base)
        elif self.type == 'grf':
            ls = self.ls
            var = self.var
            eps = self.eps
            base_dist = self._get_base_dist_grf(h_base, w_base, ls, var, eps)
        else:
            raise ValueError(f'Unknown prior type: {self.config_base_dist["type"]}')

        self.base_dist_cache = {h_base: {w_base: base_dist}}
        return base_dist

    def _get_base_dist_standard(self, h_base, w_base):
            n = h_base * w_base
            base_dist = MultivariateNormal(
                torch.zeros(n).to(device), 
                torch.eye(n).to(device)
            )
            return base_dist

    def _get_base_dist_grf(self, h_base, w_base, ls, var, eps):
        n = h_base * w_base
        x = torch.linspace(0, 1, h_base, device=device)
        y = torch.linspace(0, 1, w_base, device=device)

        coords = torch.cartesian_prod(x, y)
        coords_a = coords.repeat(n, 1)
        coords_b = coords.repeat_interleave(n, dim=0)
        coord_pairs = torch.cat((coords_a, coords_b), dim=1)

        I = torch.eye(n, device=device)
        diff_norm = (coord_pairs[:, :2] - coord_pairs[:, 2:]).norm(dim=-1)
        K = var * torch.exp(-0.5 * diff_norm **2 / ls**2).view(n, n) + eps * I
        mu = torch.zeros(n, device=device)

        base_dist = MultivariateNormal(mu, K)
        return base_dist