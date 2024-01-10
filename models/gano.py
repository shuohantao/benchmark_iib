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
        pass
    def sample(self, resolution, num_samples=1, device="cuda", **kwargs):
        grf = GaussianRF_idct(2, resolution, alpha=self.grf.alpha, tau=self.grf.tau, device=device)
        return self.G(grf.sample(num_samples).unsqueeze(-1).permute(0, 3, 1, 2))