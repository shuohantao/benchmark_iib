import torch
from modules.unet import UNet
from modules.RealNVP import RealNVP
import torch.nn.functional as F

class UnetFlow(torch.nn.Module):
    def __init__(self, base_res, context_num, *args, **kwargs) -> None:
        super().__init__(base_res, context_num, *args, **kwargs)
        self.unet = UNet(1, 2, 2)
        self.flow = RealNVP(16, 16, 4, 8)
        self.base_res = base_res
    def forward(self, x):
        b, c, h, w = x.shape
        x_low = F.interpolate(x, size=(self.base_res, self.base_res), mode='bilinear')
        x_nn = F.interpolate(x, size=(h, w), mode='nearest')
        condition = torch.tensor([x_low.shape[-1], w], dtype=torch.float).to(x.device).expand(b, 2)
        mu, sig = torch.chunk(self.unet(x_nn, condition), 2, dim=1)
        sldj = self.flow(x, mu, sig)
        return -sldj
    def sample(self, x, num_context, autoregessive=False, num_samples=-1, *args, **kwargs):
        x = x[:num_samples]
        b, c, h, w = x.shape
        x_low = F.interpolate(x, size=(self.base_res, self.base_res), mode='bilinear')
        x_nn = F.interpolate(x, size=(h, w), mode='nearest')
        condition = torch.tensor([x_low.shape[-1], w], dtype=torch.float).to(x.device).expand(b, 2)
        mu, sig = torch.chunk(self.unet(x_nn, condition), 2, dim=1)
        return self.flow.sample(mu, sig)