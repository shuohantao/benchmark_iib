from torch.distributions.normal import Normal
import torch
import numpy as np
import torch_dct as dct

class GRF(object):
    def __init__(self, alpha, tau) -> None:
        self.alpha = alpha
        self.tau = tau
        self.prior = Normal(0, 1)
        self.pen_coeff = 1e5
    def calc_likelihood(self, x, modes, penalize =False):
        x = dct.dct_2d(x, norm='ortho')
        Ln = x.shape[-1]
        k = torch.arange(Ln).to(x.device)
        K1,K2 = torch.meshgrid(k,k)
        K1 = K1.to(x.device)
        K2 = K2.to(x.device)
        # Define the (square root of) eigenvalues of the covariance operator
        C_base = (np.pi**2)*(K1**2 + K2**2)+self.tau**2
        C = torch.pow(C_base,-self.alpha/2.0)
        C = (self.tau**(self.alpha-1))*C
        ldj = -0.5*self.alpha*torch.sum(torch.log(C_base)) + Ln**2*(self.alpha-1)*np.log(self.tau) + Ln**2*np.log(Ln)
        x = x/(Ln*C)
        logp = self.prior.log_prob(x)
        if penalize:
            lower_modes = logp[..., :modes, :modes].clone()
            logp[..., :modes, :modes] = -self.pen_coeff*torch.abs(lower_modes)
        prior_ldj = logp.sum()
        # prior_ldj = 0
        return prior_ldj, ldj*x.shape[0]
    def sample_scales(self, scales, num, device):
        l = scales[-1]
        k = torch.arange(l).to(device)
        K1,K2 = torch.meshgrid(k,k)
        K1 = K1.to(device)
        K2 = K2.to(device)
        C = (np.pi**2)*(K1**2 + K2**2)+self.tau**2
        C = torch.pow(C,-self.alpha/2.0)
        C = (self.tau**(self.alpha-1))*C
        x = self.prior.sample((num, 1, l, l)).to(device)
        x = x*C
        pyramids = [dct.idct_2d(x[..., :scales[0], :scales[0]]*scales[0], norm='ortho')]
        for s in range(len(scales)-1):
            s = s+1
            x_next = x[..., :scales[s], :scales[s]]*scales[s]
            pyramids.append(dct.idct_2d(x_next, norm='ortho'))
        return pyramids
    def sample(self, num, res, device):
        k = torch.arange(res).to(device)
        K1,K2 = torch.meshgrid(k,k)
        K1 = K1.to(device)
        K2 = K2.to(device)
        C = (np.pi**2)*(K1**2 + K2**2)+self.tau**2
        C = torch.pow(C,-self.alpha/2.0)
        C = (self.tau**(self.alpha-1))*C
        x = self.prior.sample((num, 1, res, res)).to(device)
        x = x*C*res
        return dct.idct_2d(x, norm='ortho')
                    


