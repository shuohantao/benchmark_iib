#modified from https://github.com/neuraloperator/GANO
import torch.nn as nn
import torch
from modules.gano_parts import *
from modules.random_fields import *

class Gano(nn.Module):
    def __init__(self, d_co_domain, modes_schedule, npad,  n_critic, default_res, lambda_grad, alpha, tau, **kwargs) -> None:
        super().__init__()
        self.D = Discriminator(1+2, d_co_domain, modes_schedule=modes_schedule, pad=npad)
        self.G = Generator(1+2, d_co_domain, modes_schedule=modes_schedule, pad=npad)
        self.n_critic = n_critic
        self.grf = GaussianRF_idct(2, default_res, alpha=alpha, tau=tau)
        self.lambda_grad = lambda_grad
        self.default_res = default_res
        self.train_iter = 1
    def calculate_gradient_penalty(self, model, real_images, fake_images, device):
        """Calculates the gradient penalty loss for GANO"""
        # Random weight term for interpolation between real and fake data
        alpha = torch.randn((real_images.size(0), 1, 1, 1), device=device)
        # Get random interpolation between real and fake data
        interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

        model_interpolates = model(interpolates)
        grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=model_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1/np.sqrt(self.default_res * self.default_res)) ** 2)
        return gradient_penalty
    def forward(self, x):
        grf = GaussianRF_idct(2, x.shape[-1], alpha=self.grf.alpha, tau=self.grf.tau, device=x.device)
        if self.train_iter % self.n_critic != 0:
            self.D.eval()
            self.G.train()
            x_syn = grf.sample(x.shape[0]).unsqueeze(-1).permute(0, 3, 1, 2).to(x.device)
            x_syn = self.G(x_syn)
            W_loss = -torch.mean(self.D(x)) + torch.mean(self.D(x_syn.detach()))

            gradient_penalty = self.calculate_gradient_penalty(self.D, x.data, x_syn.data, x.device)

            loss_D = W_loss + self.lambda_grad * gradient_penalty
            loss_G = None
        else:
            self.G.eval()
            self.D.train()
            x_syn = self.G(grf.sample(x.shape[0]).unsqueeze(-1).permute(0, 3, 1, 2).to(x.device))
            loss_G = -torch.mean(self.D(x_syn))
            loss_D = None
        self.train_iter += 1
        return loss_G, loss_D
    def sample(self, resolution, num_samples=1, device="cuda", **kwargs):
        grf = GaussianRF_idct(2, resolution, alpha=self.grf.alpha, tau=self.grf.tau, device=device)
        return self.G(grf.sample(num_samples).unsqueeze(-1).permute(0, 3, 1, 2))