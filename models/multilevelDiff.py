#modified from https://github.com/htlambley/multilevelDiff
import torch.nn.functional as F
import torch
import torch.nn as nn
from modules.priors_new import *
from modules.fno import *
from tqdm import tqdm
# huang sde code
device = 'cuda'
class MultilevelDiff(nn.Module):
    def __init__(self, prior_modes_1, prior_modes_2, prior_scale, width, **kwargs):
        super().__init__()
        self.prior = FNOprior(k1=prior_modes_1,k2=prior_modes_2, scale=prior_scale)
        self.fwd_sde = VariancePreservingSDE(self.prior, alpha_min=0.1, alpha_max=20.0, T=1. )
        self.model = FNO2d(prior_modes_2, prior_modes_2, width)
        self.rev_sde = PluginReverseSDE(self.prior, self.fwd_sde, self.model, 1., vtype='Rademacher', debias=False).to(device)
    def forward(self, x):
        loss = self.rev_sde.dsm(x).mean()
        return loss
    def sample(self, resolution, num_samples=1, device="cuda", input_channels=1, num_steps=200, **kwargs):
        sde = self.rev_sde
        delta = sde.T / num_steps
   
        for l in tqdm(range(100)):
            y0 = sde.prior.sample([num_samples, input_channels, resolution, resolution])
            ts = torch.linspace(0, 1, num_steps + 1).to(y0) * sde.T
            ones = torch.ones(num_samples, 1, 1, 1).to(y0)

            with torch.no_grad():
                for i in range(num_steps):
                    mu = sde.mu(ones * ts[i], y0, lmbd = 0.)
                    sigma = sde.sigma(ones * ts[i], y0, lmbd = 0.)
                    epsilon = sde.prior.sample(y0.shape)
                    y0 = y0 + delta * mu + (delta ** 0.5) * sigma * epsilon
        return y0

class VariancePreservingSDE(torch.nn.Module):
    """
    Implementation of the variance preserving SDE proposed by Song et al. 2021
    See eq (32-33) of https://openreview.net/pdf?id=PxTIG12RRHS
    """
    def __init__(self, prior, alpha_min=0.1, alpha_max=20.0, T=1.0, t_epsilon=0.001):
        super().__init__()
        self.prior = prior
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.T = T
        self.t_epsilon = t_epsilon 
        
    def alpha(self, t):
        return self.alpha_min + (self.alpha_max-self.alpha_min)*t

    def mean_weight(self, t): #tilde alpha in paper
        return torch.exp(-0.25 * t**2 * (self.alpha_max-self.alpha_min) - 0.5 * t * self.alpha_min)

    def var_weight(self, t): #tilde beta in paper
        return 1. - torch.exp(-0.5 * t**2 * (self.alpha_max-self.alpha_min) - t * self.alpha_min)

    def f(self, t, y):
        return - 0.5 * self.alpha(t) * y

    def g(self, t, y):
        alpha_t = self.alpha(t)
        return torch.ones_like(y) * alpha_t**0.5

    def sample(self, t, y0, return_noise=False):
        """
        sample yt | y0
        if return_noise=True, also return std and g for reweighting the denoising score matching loss
        """
        mu = self.mean_weight(t) * y0  #mean = tilde alpha * y0
        std_weight = self.var_weight(t) ** 0.5        
        eta = self.prior.sample(y0.shape) 
        yt = eta * std_weight + mu
        if not return_noise:
            return yt
        else:
            return yt, eta, std_weight, self.g(t, yt)


class PluginReverseSDE(torch.nn.Module):
    """
    inverting a given base sde with drift `f` and diffusion `g`, and an inference sde's drift `a` by
    f <- g a - f
    g <- g
    (time is inverted)
    https://github.com/CW-Huang/sdeflow-light
    """

    def __init__(self, prior,base_sde, drift_a, T, vtype='rademacher', debias=False):
        super().__init__()
        self.base_sde = base_sde
        self.prior = prior
        self.a = drift_a
        self.T = T
        self.vtype = vtype
        self.debias = debias
    
    # Drift
    def mu(self, t, y, lmbd=0.):
        a = self.a(y, self.T - t.squeeze())
        return (1. - 0.5 * lmbd) * self.prior.Q_g2_s(self.base_sde.g(self.T-t, y),a) - \
               self.base_sde.f(self.T - t, y)

    # Diffusion
    def sigma(self, t, y, lmbd=0.):
        return (1. - lmbd) ** 0.5 * self.base_sde.g(self.T-t, y)

    @torch.enable_grad()
    def dsm(self, x):
        """
        denoising score matching loss
        """ 
        t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        y, target, std_weight, g = self.base_sde.sample(t_, x, return_noise=True)
        target = target.to(device) #eta

        a = self.a(y, t_.squeeze())
        score = a*std_weight / g  #a =(Q*(y-mu)*g)/(std_weight*Q)

        return ((score+target)**2).view(x.size(0), -1).sum(1, keepdim=False) / 2 

