from torch.distributions.normal import Normal
import torch.nn as nn
import torch
#corrected implementation of binned MoG from https://arxiv.org/pdf/2103.15813.pdf
class MoG(object):
    def __init__(self, bin_centres):
        self.prior = Normal(0, 1)
        self.bin_centres = bin_centres
    def log_prob(self, tau, mu, sigs, x):
        mean = self.bin_centres + mu
        x_bin_norm = (x - mean) / sigs
        log_probs = self.prior.log_prob(x_bin_norm)
        log_mix = torch.log_softmax(tau, dim=-1)
        return torch.logsumexp(log_probs + log_mix, dim=-1)
class MoGNearest(object):
    def __init__(self, bin_centres):
        self.prior = Normal(0, 1)
        self.bin_centres = bin_centres
    def log_prob(self, tau, mu, sigs, x):
        b_id = torch.min(torch.abs(self.bin_centres - x), dim=-1).indices
        mu = mu[..., b_id]
        mean = (self.bin_centres[..., b_id] + mu).unsqueeze(-1)
        sigs = sigs[..., b_id].unsqueeze(-1)
        x_bin_norm = (x - mean) / sigs
        log_probs = self.prior.log_prob(x_bin_norm)
        log_mix = torch.log_softmax(tau, dim=-1)
        return log_mix[..., b_id].unsqueeze(-1) + log_probs

