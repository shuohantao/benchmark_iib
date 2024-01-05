import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .neurop import *
class CNN_Linear(nn.Module):
    def __init__(self, c_in, shape, c_hidden=32, c_out=-1, kernel_size = 3, padding=0, **kwargs):
        super().__init__()
        self.shape = shape
        self.c_out = c_out = c_out if c_out > 0 else 2 * c_in
        self.dim_through = np.prod(shape[-2:]) * c_out
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1),
            nn.Softplus(),
            nn.BatchNorm2d(c_hidden),
            nn.Conv2d(c_hidden, c_hidden, kernel_size=3, padding=1),
            nn.Softplus(),
            nn.BatchNorm2d(c_hidden),
            nn.Conv2d(c_hidden, c_out, kernel_size=3, padding=1),
            torch.nn.Flatten(),
            torch.nn.Linear(self.dim_through, self.dim_through//2),
            nn.Softplus(),
            torch.nn.Linear(self.dim_through//2, self.dim_through//2),
            nn.Softplus(),
            torch.nn.Linear(self.dim_through//2, self.dim_through),
        )
    def forward(self, x, **kwargs):
        st = self.net(x)
        s, t = st.chunk(2, dim=1)
        s = s.view(-1, self.shape[1], *list(self.shape[-2:]))
        t = t.view(-1, self.shape[1], *list(self.shape[-2:]))
        return torch.cat((s, t), dim=1)
    
class Rec_Res_CNN(nn.Module):
    def __init__(self, c_in, shape, c_hidden=32, c_out=-1):
        super().__init__()
        self.shape = shape
        self.c_out = c_out = c_out if c_out > 0 else 2 * c_in
        self.dim_through = np.prod(shape[-2:]) * c_out
        self.res_blocks = nn.ModuleList([nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(c_hidden),
            nn.Conv2d(c_hidden, c_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(c_hidden),
            nn.Conv2d(c_hidden, c_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(c_hidden),
            nn.Conv2d(c_hidden, c_in, kernel_size=3, padding=1)) for i in range(3)]
            +[nn.Sequential(nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(c_hidden),
            nn.Conv2d(c_hidden, c_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(c_hidden),
            nn.Conv2d(c_hidden, c_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(c_hidden),
            nn.Conv2d(c_hidden, c_out, kernel_size=3, padding=1))])
    def forward(self, x, **kwargs):
        for i in self.res_blocks[:-1]:
            x = x + i(x)
        return self.res_blocks[-1](x)
    
class CNN(nn.Module):
    def __init__(self, c_in, c_out=2, n_hidden = 32, kernel_size = 1, padding=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, n_hidden * c_in, kernel_size=kernel_size, padding=padding),
            nn.Softplus(),
            nn.BatchNorm2d(n_hidden * c_in),
            nn.Conv2d(n_hidden * c_in, n_hidden * c_in, kernel_size=kernel_size, padding=padding),
            nn.Softplus(),
            nn.BatchNorm2d(n_hidden * c_in),
            nn.Conv2d(n_hidden * c_in, c_out, kernel_size=kernel_size, padding=padding),
        )
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()
    def forward(self, x):
        st = self.net(x)
        s, t = st.chunk(2, dim=1)
        return s, t

class FCL(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.dim_through = np.prod(shape[-3:])
        self.shape = shape
        self.scale = nn.Sequential(torch.nn.Flatten(),
                                torch.nn.Linear(self.dim_through, self.dim_through//2),
                                nn.Tanh(),
                                torch.nn.Linear(self.dim_through//2, self.dim_through//2),
                                nn.Tanh(),
                                torch.nn.Linear(self.dim_through//2, np.prod(shape[-2:])),
                                )
        self.bias = nn.Sequential(torch.nn.Flatten(),
                                torch.nn.Linear(self.dim_through, self.dim_through//2),
                                nn.Tanh(),
                                torch.nn.Linear(self.dim_through//2, self.dim_through//2),
                                nn.Tanh(),
                                torch.nn.Linear(self.dim_through//2, np.prod(shape[-2:]))
                                )
    def forward(self, x):
        s = self.scale(x)
        t = self.bias(x)
        s = s.view(-1, 1, *list(self.shape[-2:]))
        t = t.view(-1, 1, *list(self.shape[-2:]))
        return s, t

class NeuropCNN(nn.Module):
    def __init__(self, device=torch.device('cuda'), c_in = 1, c_hidden = 64, c_out = -1, default_shape = 28, linear=True, kernel_shapes = [3]*3, residual=None, transformer=False, mode="interpolate", fourier=False):
        super().__init__()
        default_shape = torch.Size([64, 1, default_shape, default_shape])
        self.default_x = default_shape[-1]
        self.default_y = default_shape[-2]
        self.default_ks = kernel_shapes
        self.device = device
        self.linear_dim = np.prod(default_shape[-2:])
        self.linear = linear
        self.transformer = transformer
        assert mode in ["interpolate", "extrapolate"], "mode must be either interpolate or extrapolate"
        self.mode = mode
        c_out = c_out if c_out > 0 else 2 if c_in == 1 else 2 * (c_in - 1)
        self.c_out = c_out
        self.residual = residual
        if fourier:
            self.krn_operators = nn.ModuleList([NeurOpKrnMLPFF(c_in = c_in, c_out = c_hidden)]+[NeurOpKrnMLPFF(c_hidden, c_hidden) for _ in range(len(kernel_shapes)-2)]+[NeurOpKrnMLPFF(c_hidden, c_out = c_out)])
        else:
            self.krn_operators = nn.ModuleList([NeurOpKrnMLP(c_in = c_in, c_out = c_hidden)]+[NeurOpKrnMLP(c_hidden, c_hidden) for _ in range(len(kernel_shapes)-2)]+[NeurOpKrnMLP(c_hidden, c_out = c_out)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(c_hidden)) for _ in range(len(kernel_shapes) - 1)] + [nn.Parameter(torch.zeros(c_out))])
        self.act_list = nn.ModuleList([
            nn.Sequential(
            nn.Softplus(),
            nn.BatchNorm2d(c_hidden),
            ) for _ in range(len(kernel_shapes) - 1)])
        if linear:
            self.fcl = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.linear_dim*c_out, self.linear_dim*c_out//4),
                nn.Softplus(),
                nn.Linear(self.linear_dim*c_out//4, self.linear_dim*c_out//4),
                nn.Softplus(),
                nn.Linear(self.linear_dim*c_out//4, self.linear_dim*c_out),)
        if transformer:
            base_dim = int(self.linear_dim**0.5) + int(self.linear_dim**0.5)%2
            self.tfm = NeurOpKrnTfm(default_shape[1]*base_dim**2//2, default_shape[1]*base_dim**2//2, nhead=1, nhid=2048)
    def _get_params(self, shape, mode="inteprolate"):
        
        ratio = shape[-1]/self.default_x
        idc = []
        props = []
        if mode == "interpolate":
            for i in self.default_ks:
                k = round(ratio * i) if round(ratio * i) % 2 != 0 else round(ratio * i) + 1
                props.append((k - 1)//2)
                X, Y = torch.meshgrid(torch.tensor([1/(2 * k) + i / k for i in range(k)]), torch.tensor([1/(2 * k) + i / k for i in range(k)]), indexing='xy')
                idc.append(torch.stack([X, Y], dim=2).view(-1, 2))
        else:
            for i in self.default_ks:
                k = round(ratio * i) if round(ratio * i) % 2 != 0 else round(ratio * i) + 1
                props.append((k - 1)//2)
                X, Y = torch.meshgrid(torch.tensor([0.5 + i for i in range(k)]), torch.tensor([0.5 + i for i in range(k)]), indexing='xy')
                idc.append(torch.stack([X, Y], dim=2).view(-1, 2))
        return idc + props
    def _create_kernel(self, neurop, coord, k_shape):
        kernel = neurop(coord.to(self.device), k_shape)
        return kernel  
        
    def forward(self, x, shape):       
        params = self._get_params(shape, mode=self.mode) 
        res_counter = 0
        res_con = None
        for i, o in enumerate(self.krn_operators):
            kernel = self._create_kernel(o, params[i], int(params[i].shape[0]**0.5))
            x = F.conv2d(input=x, weight=kernel, stride=1, padding=params[len(self.krn_operators)+i], bias=self.bias[i].view(-1))
            if self.residual is not None and i < len(self.krn_operators) - self.residual:
                if res_con is not None:
                    x += res_con
            if i < len(self.krn_operators) - 1:
                x = self.act_list[i](x)
            if self.residual is not None and i < len(self.krn_operators) - self.residual:
                res_counter += 1
                if res_counter == self.residual:
                    res_counter = 0
                    res_con = x
        if self.linear:
            if np.prod(x.shape[-2:]) > self.linear_dim:
                x = F.adaptive_avg_pool2d(x, (int(self.linear_dim**0.5), )*2)
                x = self.fcl(x).view(-1, 2*shape[1], int(self.linear_dim**0.5), int(self.linear_dim**0.5))
                x = F.interpolate(x, size=tuple(shape[-2:]), mode='bilinear', align_corners=False)
            elif np.prod(x.shape[-2:]) < self.linear_dim:
                x = F.interpolate(x, size=(int(self.linear_dim**0.5), )*2, mode='bilinear', align_corners=False)
                x = self.fcl(x).view(-1, 2*shape[1], int(self.linear_dim**0.5), int(self.linear_dim**0.5))
                x = F.adaptive_avg_pool2d(x, tuple(shape[-2:]))
            else:
                x = self.fcl(x).view(-1, 2*shape[1], int(self.linear_dim**0.5), int(self.linear_dim**0.5))
        if self.transformer:
            base_dim = int(self.linear_dim**0.5) + int(self.linear_dim**0.5)%2
            print(base_dim, x.shape)
            if np.prod(x.shape[-2:]) > self.linear_dim:
                x = F.adaptive_avg_pool2d(x, (base_dim, )*2)
                x = x.reshape(-1, 2*shape[1], base_dim//2, 2, base_dim//2, 2).permute(2, 3, 4, 0, 1, 5).reshape(4, -1, np.prod(x.shape[-3:])//4)
                x = self.tfm(x).reshape(base_dim//2, 2, base_dim//2, 2, -1, 2*shape[1]).permute(4, 5, 0, 2, 1, 3).reshape(-1, 2*shape[1], base_dim, base_dim)
                x = F.interpolate(x, size=tuple(shape[-2:]), mode='bilinear', align_corners=False)
            elif np.prod(x.shape[-2:]) < self.linear_dim:
                x = F.interpolate(x, size=(base_dim, )*2, mode='bilinear', align_corners=False)
                x = x.reshape(-1, 2*shape[1], base_dim//2, 2, base_dim//2, 2).permute(2, 3, 4, 0, 1, 5).reshape(4, -1, np.prod(x.shape[-3:])//4)
                x = self.tfm(x).reshape(base_dim//2, 2, base_dim//2, 2, -1, 2*shape[1]).permute(4, 5, 0, 2, 1, 3).reshape(-1, 2*shape[1], base_dim, base_dim)
                x = F.adaptive_avg_pool2d(x, tuple(shape[-2:]))
            else:
                x = x.reshape(-1, 2*shape[1], base_dim//2, 2, base_dim//2, 2).permute(2, 3, 4, 0, 1, 5).reshape(4, -1, np.prod(x.shape[-3:])//4)
                print(x.shape)
                x = self.tfm(x).reshape(base_dim//2, 2, base_dim//2, 2, -1, 2*shape[1]).permute(4, 5, 0, 2, 1, 3).reshape(-1, 2*shape[1], base_dim, base_dim)
        return x

class NeuropCNNTransformer(nn.Module):
    def __init__(self, device, kernel_shapes, c_in = 1, c_hidden = 64, c_out = -1, default_shape = torch.Size([64, 1, 28, 28]), channel_schedule = None, linear=False, residual=None):
        super().__init__()
        self.default_x = default_shape[-1]
        self.default_y = default_shape[-2]
        self.default_c = c_in
        self.default_ks = kernel_shapes
        self.device = device
        self.linear_dim = np.prod(default_shape[-2:])
        self.linear = linear
        c_out = c_out if c_out > 0 else 2 * c_in
        self.c_out = c_out
        self.c_hidden = c_hidden
        self.residual = residual
        if not channel_schedule:
            self.channel_schedule = [c_hidden]*(len(kernel_shapes) - 1) + [c_out]
        else:
            self.channel_schedule = channel_schedule
        self.krn_operators = NeurOpKrnTfm(2, (len(self.default_ks)-2)*c_hidden**2 + c_hidden*c_in + c_hidden*c_out, nhead=2, nhid=2048, dropout=0.2, nlayers=6)
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(c_hidden)) for _ in range(len(kernel_shapes) - 1)] + [nn.Parameter(torch.zeros(c_out))])
        self.act_list = nn.ModuleList([
            nn.Sequential(
            nn.Softplus(),
            nn.BatchNorm2d(c_hidden),
            ) for _ in range(len(kernel_shapes) - 1)])
        if linear:
            self.fcl = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.linear_dim*c_out, self.linear_dim*c_out//4),
                nn.Softplus(),
                nn.Linear(self.linear_dim*c_out//4, self.linear_dim*c_out//4),
                nn.Softplus(),
                nn.Linear(self.linear_dim*c_out//4, self.linear_dim*c_out),)
    def _get_params(self, shape):
        ratio = shape[-1]/self.default_x
        i = self.default_ks[0]
        k = round(ratio * i) if round(ratio * i) % 2 != 0 else round(ratio * i) + 1
        padding = (k - 1)//2
        X, Y = torch.meshgrid(torch.tensor([1/(2 * k) + i / k for i in range(k)]), torch.tensor([1/(2 * k) + i / k for i in range(k)]), indexing='xy')
        idc = torch.stack([X, Y], dim=2).view(-1, 2)
        return idc, padding
        
    def forward(self, x, shape):       
        idc, padding = self._get_params(shape) 
        res_counter = 0
        res_con = None
        coords = idc.unsqueeze(1).to(self.device)
        kernels = self.krn_operators(coords).permute(2, 1, 0)
        prev_chan = self.default_c
        chan_count = 0
        for i, o in enumerate(self.channel_schedule):
            num = prev_chan * o
            k_slice = kernels[chan_count:chan_count+num]
            k = k_slice.view(o, prev_chan, int(idc.shape[0]**0.5), int(idc.shape[0]**0.5))
            chan_count += num
            prev_chan = o
            x = F.conv2d(input=x, weight=k, stride=1, padding=padding, bias=self.bias[i].view(-1))
            if self.residual is not None and i < len(self.default_ks) - self.residual:
                if res_con is not None:
                    x += res_con
            if i < len(self.default_ks) - 1:
                x = self.act_list[i](x)
            if self.residual is not None and i < len(self.default_ks) - self.residual:
                res_counter += 1
                if res_counter == self.residual:
                    res_counter = 0
                    res_con = x
        if self.linear:
            if np.prod(x.shape[-2:]) > self.linear_dim:
                x = F.adaptive_avg_pool2d(x, (int(self.linear_dim**0.5), )*2)
                x = self.fcl(x).view(-1, 2*shape[1], int(self.linear_dim**0.5), int(self.linear_dim**0.5))
                x = F.interpolate(x, size=tuple(shape[-2:]), mode='bilinear', align_corners=False)
            elif np.prod(x.shape[-2:]) < self.linear_dim:
                x = F.interpolate(x, size=(int(self.linear_dim**0.5), )*2, mode='bilinear', align_corners=False)
                x = self.fcl(x).view(-1, 2*shape[1], int(self.linear_dim**0.5), int(self.linear_dim**0.5))
                x = F.adaptive_avg_pool2d(x, tuple(shape[-2:]))
            else:
                x = self.fcl(x).view(-1, 2*shape[1], int(self.linear_dim**0.5), int(self.linear_dim**0.5))
        return x

class NeuropCNNConditioned(nn.Module):
    def __init__(self, device=torch.device('cuda'), c_in = 1, c_hidden = 64, c_out = -1, default_shape = 28, linear=False, kernel_shapes = [3]*3, residual=None, transformer=False, mode="interpolate", fourier=False):
        super().__init__()
        default_shape = torch.Size([64, 1, default_shape, default_shape])
        self.default_x = default_shape[-1]
        self.default_y = default_shape[-2]
        self.default_ks = kernel_shapes
        self.device = device
        self.linear_dim = np.prod(default_shape[-2:])
        self.linear = linear
        self.transformer = transformer
        assert mode in ["interpolate", "extrapolate"], "mode must be either interpolate or extrapolate"
        self.mode = mode
        c_out = c_out if c_out > 0 else 2 if c_in == 1 else 2 * (c_in - 1)
        self.c_out = c_out
        self.residual = residual
        if fourier:
            self.krn_operators = nn.ModuleList([NeurOpKrnMLPFF(c_in = c_in, c_out = c_hidden, conditioned=True)]+[NeurOpKrnMLPFF(c_hidden, c_hidden, conditioned=True) for _ in range(len(kernel_shapes)-2)]+[NeurOpKrnMLPFF(c_hidden, c_out = c_out, conditioned=True)])
        else:
            self.krn_operators = nn.ModuleList([NeurOpKrnMLP(c_in = c_in, c_out = c_hidden)]+[NeurOpKrnMLP(c_hidden, c_hidden) for _ in range(len(kernel_shapes)-2)]+[NeurOpKrnMLP(c_hidden, c_out = c_out)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(c_hidden)) for _ in range(len(kernel_shapes) - 1)] + [nn.Parameter(torch.zeros(c_out))])
        self.act_list = nn.ModuleList([
            nn.Sequential(
            nn.Softplus(),
            nn.BatchNorm2d(c_hidden),
            ) for _ in range(len(kernel_shapes) - 1)])
        if linear:
            self.fcl = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.linear_dim*c_out, self.linear_dim*c_out//4),
                nn.Softplus(),
                nn.Linear(self.linear_dim*c_out//4, self.linear_dim*c_out//4),
                nn.Softplus(),
                nn.Linear(self.linear_dim*c_out//4, self.linear_dim*c_out),)
        if transformer:
            base_dim = int(self.linear_dim**0.5) + int(self.linear_dim**0.5)%2
            self.tfm = NeurOpKrnTfm(default_shape[1]*base_dim**2//2, default_shape[1]*base_dim**2//2, nhead=1, nhid=2048)
    def _get_params(self, shape, mode="inteprolate"):
        
        ratio = shape[-1]/self.default_x
        idc = []
        props = []
        if mode == "interpolate":
            for i in self.default_ks:
                k = round(ratio * i) if round(ratio * i) % 2 != 0 else round(ratio * i) + 1
                props.append((k - 1)//2)
                X, Y = torch.meshgrid(torch.tensor([1/(2 * k) + i / k for i in range(k)]), torch.tensor([1/(2 * k) + i / k for i in range(k)]), indexing='xy')
                idc.append(torch.stack([X, Y], dim=2).view(-1, 2))
        else:
            for i in self.default_ks:
                k = round(ratio * i) if round(ratio * i) % 2 != 0 else round(ratio * i) + 1
                props.append((k - 1)//2)
                X, Y = torch.meshgrid(torch.tensor([0.5 + i for i in range(k)]), torch.tensor([0.5 + i for i in range(k)]), indexing='xy')
                idc.append(torch.stack([X, Y], dim=2).view(-1, 2))
        return idc + props
    def _create_kernel(self, neurop, coord, k_shape, condition):
        kernel = neurop(coord.to(self.device), k_shape, condition=condition)
        return kernel  
        
    def forward(self, x, shape, condition):       
        params = self._get_params(shape, mode=self.mode) 
        res_counter = 0
        res_con = None
        for i, o in enumerate(self.krn_operators):
            kernel = self._create_kernel(o, params[i], int(params[i].shape[0]**0.5), condition=condition)
            x = F.conv2d(input=x, weight=kernel, stride=1, padding=params[len(self.krn_operators)+i], bias=self.bias[i].view(-1))
            if self.residual is not None and i < len(self.krn_operators) - self.residual:
                if res_con is not None:
                    x += res_con
            if i < len(self.krn_operators) - 1:
                x = self.act_list[i](x)
            if self.residual is not None and i < len(self.krn_operators) - self.residual:
                res_counter += 1
                if res_counter == self.residual:
                    res_counter = 0
                    res_con = x
        if self.linear:
            if np.prod(x.shape[-2:]) > self.linear_dim:
                x = F.adaptive_avg_pool2d(x, (int(self.linear_dim**0.5), )*2)
                x = self.fcl(x).view(-1, 2*shape[1], int(self.linear_dim**0.5), int(self.linear_dim**0.5))
                x = F.interpolate(x, size=tuple(shape[-2:]), mode='bilinear', align_corners=False)
            elif np.prod(x.shape[-2:]) < self.linear_dim:
                x = F.interpolate(x, size=(int(self.linear_dim**0.5), )*2, mode='bilinear', align_corners=False)
                x = self.fcl(x).view(-1, 2*shape[1], int(self.linear_dim**0.5), int(self.linear_dim**0.5))
                x = F.adaptive_avg_pool2d(x, tuple(shape[-2:]))
            else:
                x = self.fcl(x).view(-1, 2*shape[1], int(self.linear_dim**0.5), int(self.linear_dim**0.5))
        if self.transformer:
            base_dim = int(self.linear_dim**0.5) + int(self.linear_dim**0.5)%2
            print(base_dim, x.shape)
            if np.prod(x.shape[-2:]) > self.linear_dim:
                x = F.adaptive_avg_pool2d(x, (base_dim, )*2)
                x = x.reshape(-1, 2*shape[1], base_dim//2, 2, base_dim//2, 2).permute(2, 3, 4, 0, 1, 5).reshape(4, -1, np.prod(x.shape[-3:])//4)
                x = self.tfm(x).reshape(base_dim//2, 2, base_dim//2, 2, -1, 2*shape[1]).permute(4, 5, 0, 2, 1, 3).reshape(-1, 2*shape[1], base_dim, base_dim)
                x = F.interpolate(x, size=tuple(shape[-2:]), mode='bilinear', align_corners=False)
            elif np.prod(x.shape[-2:]) < self.linear_dim:
                x = F.interpolate(x, size=(base_dim, )*2, mode='bilinear', align_corners=False)
                x = x.reshape(-1, 2*shape[1], base_dim//2, 2, base_dim//2, 2).permute(2, 3, 4, 0, 1, 5).reshape(4, -1, np.prod(x.shape[-3:])//4)
                x = self.tfm(x).reshape(base_dim//2, 2, base_dim//2, 2, -1, 2*shape[1]).permute(4, 5, 0, 2, 1, 3).reshape(-1, 2*shape[1], base_dim, base_dim)
                x = F.adaptive_avg_pool2d(x, tuple(shape[-2:]))
            else:
                x = x.reshape(-1, 2*shape[1], base_dim//2, 2, base_dim//2, 2).permute(2, 3, 4, 0, 1, 5).reshape(4, -1, np.prod(x.shape[-3:])//4)
                print(x.shape)
                x = self.tfm(x).reshape(base_dim//2, 2, base_dim//2, 2, -1, 2*shape[1]).permute(4, 5, 0, 2, 1, 3).reshape(-1, 2*shape[1], base_dim, base_dim)
        return x
    
class ConcatELU(nn.Module):
    """
    Activation function that applies ELU in both direction (inverted and plain).
    Allows non-linearity while providing strong gradients for any input (important for final convolution)
    """

    def forward(self, x):
        return torch.cat([F.elu(x), F.elu(-x)], dim=1)


class LayerNormChannels(nn.Module):

    def __init__(self, c_in, eps=1e-5):
        """
        This module applies layer norm across channels in an image.
        Inputs:
            c_in - Number of channels of the input
            eps - Small constant to stabilize std
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, c_in, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, c_in, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        y = y * self.gamma + self.beta
        return y


class GatedConv(nn.Module):

    def __init__(self, c_in, c_hidden):
        """
        This module applies a two-layer convolutional ResNet block with input gate
        Inputs:
            c_in - Number of channels of the input
            c_hidden - Number of hidden dimensions we want to model (usually similar to c_in)
        """
        super().__init__()
        self.net = nn.Sequential(
            ConcatELU(),
            nn.Conv2d(2*c_in, c_hidden, kernel_size=3, padding=1),
            ConcatELU(),
            nn.Conv2d(2*c_hidden, 2*c_in, kernel_size=1)
        )

    def forward(self, x):
        out = self.net(x)
        val, gate = out.chunk(2, dim=1)
        return x + val * torch.sigmoid(gate)


class GatedConvNet(nn.Module):

    def __init__(self, c_in, c_hidden=32, c_out=-1, num_layers=3):
        """
        Module that summarizes the previous blocks to a full convolutional neural network.
        Inputs:
            c_in - Number of input channels
            c_hidden - Number of hidden dimensions to use within the network
            c_out - Number of output channels. If -1, 2 times the input channels are used (affine coupling)
            num_layers - Number of gated ResNet blocks to apply
        """
        super().__init__()
        c_out = c_out if c_out > 0 else 2 * c_in
        layers = []
        layers += [nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1)]
        for layer_index in range(num_layers):
            layers += [GatedConv(c_hidden, c_hidden),
                       LayerNormChannels(c_hidden)]
        layers += [ConcatELU(),
                   nn.Conv2d(2*c_hidden, c_out, kernel_size=3, padding=1)]
        self.nn = nn.Sequential(*layers)

        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, x):
        return self.nn(x)
    
class FCL(nn.Module):
    def __init__(self, n_in, n_hid, n_layers):
        super().__init__()
        modules = [nn.Linear(n_in, n_hid), nn.ELU()]
        for _ in range(n_layers-2):
            modules += [nn.Linear(n_hid, n_hid), nn.ELU()]
        modules += [nn.Linear(n_hid, n_in*2)]
        self.net = nn.Sequential(*modules)
    def forward(self, x):
        return self.net(x)