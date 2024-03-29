#modified from https://github.com/neuraloperator/GANO
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpectralConv2dUno(nn.Module):
    def __init__(self, in_channels, out_channels, modes1 = None, modes2 = None):
        super(SpectralConv2dUno, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.scale = (1 / (2*in_channels))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * (torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))
        self.weights2 = nn.Parameter(self.scale * (torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, dim1 = None, dim2 = None):
        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  self.dim1, self.dim2//2 + 1 , dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(self.dim1, self.dim2))
        return x
    

class pointwise_op(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(pointwise_op,self).__init__()
        self.conv = nn.Conv2d(int(in_channel), int(out_channel), 1)


    def forward(self,x, dim1 = None, dim2 = None):
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
        x_out = self.conv(x)
        x_out = torch.nn.functional.interpolate(x_out, size = (dim1, dim2),mode = 'bicubic',align_corners=True)
        return x_out

class Generator(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain, modes_schedule, pad = 0, factor = 3/4):
        super(Generator, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.in_d_co_domain = in_d_co_domain # input channel
        self.d_co_domain = d_co_domain 
        self.factor = factor
        self.padding = pad  # pad the domain if input is non-periodic

        # self.fc0 = nn.Linear(self.in_d_co_domain, self.d_co_domain) # input channel is 3: (a(x, y), x, y)

        modes_schedule = modes_schedule+modes_schedule[-1::-1]

        self.fc0 = nn.Conv2d(self.in_d_co_domain, self.d_co_domain, 1)

        self.conv0 = SpectralConv2dUno(self.d_co_domain, 2*factor*self.d_co_domain, modes_schedule[0], modes_schedule[0])

        self.conv1 = SpectralConv2dUno(2*factor*self.d_co_domain, 4*factor*self.d_co_domain, modes_schedule[1], modes_schedule[1])

        self.conv2 = SpectralConv2dUno(4*factor*self.d_co_domain, 8*factor*self.d_co_domain, modes_schedule[2], modes_schedule[2])
        
        self.conv2_1 = SpectralConv2dUno(8*factor*self.d_co_domain, 16*factor*self.d_co_domain, modes_schedule[3], modes_schedule[3])
        
        self.conv2_9 = SpectralConv2dUno(16*factor*self.d_co_domain, 8*factor*self.d_co_domain, modes_schedule[4], modes_schedule[4])
        

        self.conv3 = SpectralConv2dUno(16*factor*self.d_co_domain, 4*factor*self.d_co_domain, modes_schedule[5], modes_schedule[5])

        self.conv4 = SpectralConv2dUno(8*factor*self.d_co_domain, 2*factor*self.d_co_domain, modes_schedule[6], modes_schedule[6])

        self.conv5 = SpectralConv2dUno(4*factor*self.d_co_domain, self.d_co_domain, modes_schedule[7], modes_schedule[7]) # will be reshaped

        self.w0 = pointwise_op(self.d_co_domain,2*factor*self.d_co_domain) #
        
        self.w1 = pointwise_op(2*factor*self.d_co_domain, 4*factor*self.d_co_domain) #
        
        self.w2 = pointwise_op(4*factor*self.d_co_domain, 8*factor*self.d_co_domain) #
        
        self.w2_1 = pointwise_op(8*factor*self.d_co_domain, 16*factor*self.d_co_domain)
        
        self.w2_9 = pointwise_op(16*factor*self.d_co_domain, 8*factor*self.d_co_domain)
        
        self.w3 = pointwise_op(16*factor*self.d_co_domain, 4*factor*self.d_co_domain) #
        
        self.w4 = pointwise_op(8*factor*self.d_co_domain, 2*factor*self.d_co_domain)
        
        self.w5 = pointwise_op(4*factor*self.d_co_domain, self.d_co_domain) # will be reshaped

        self.fc1 = nn.Conv2d(2*self.d_co_domain, 4*self.d_co_domain, 1)
        self.fc2 = nn.Conv2d(4*self.d_co_domain, 1, 1)

    def forward(self, x):

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        
        x_fc0 = self.fc0(x)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = F.pad(x_fc0, [0, self.padding, 0, self.padding])
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]
        
        x1_c0 = self.conv0(x_fc0,int(D1*self.factor),int(D2*self.factor))
        x2_c0 = self.w0(x_fc0,int(D1*self.factor),int(D2*self.factor))
        x_c0 = x1_c0 + x2_c0
        x_c0 = F.gelu(x_c0)
        #print(x.shape)

        x1_c1 = self.conv1(x_c0 ,D1//2,D2//2)
        x2_c1 = self.w1(x_c0 ,D1//2,D2//2)
        x_c1 = x1_c1 + x2_c1
        x_c1 = F.gelu(x_c1)
        #print(x.shape)

        x1_c2 = self.conv2(x_c1 ,D1//4,D2//4)
        x2_c2 = self.w2(x_c1 ,D1//4,D2//4)
        x_c2 = x1_c2 + x2_c2
        x_c2 = F.gelu(x_c2 )
        #print(x.shape)
        
        x1_c2_1 = self.conv2_1(x_c2,D1//8,D2//8)
        x2_c2_1 = self.w2_1(x_c2,D1//8,D2//8)
        x_c2_1 = x1_c2_1 + x2_c2_1
        x_c2_1 = F.gelu(x_c2_1)
        
        x1_c2_9 = self.conv2_9(x_c2_1,D1//4,D2//4)
        x2_c2_9 = self.w2_9(x_c2_1,D1//4,D2//4)
        x_c2_9 = x1_c2_9 + x2_c2_9
        x_c2_9 = F.gelu(x_c2_9)
        x_c2_9 = torch.cat([x_c2_9, x_c2], dim=1) 

        x1_c3 = self.conv3(x_c2_9,D1//2,D2//2)
        x2_c3 = self.w3(x_c2_9,D1//2,D2//2)
        x_c3 = x1_c3 + x2_c3
        x_c3 = F.gelu(x_c3)
        x_c3 = torch.cat([x_c3, x_c1], dim=1)

        x1_c4 = self.conv4(x_c3,int(D1*self.factor),int(D2*self.factor))
        x2_c4 = self.w4(x_c3,int(D1*self.factor),int(D2*self.factor))
        x_c4 = x1_c4 + x2_c4
        x_c4 = F.gelu(x_c4)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)

        x1_c5 = self.conv5(x_c4,D1,D2)
        x2_c5 = self.w5(x_c4,D1,D2)
        x_c5 = x1_c5 + x2_c5
        x_c5 = F.gelu(x_c5)
        

        x_c5 = torch.cat([x_c5, x_fc0], dim=1)
        if self.padding!=0:
            x_c5 = x_c5[..., :-self.padding, :-self.padding]
        
        x_fc1 = self.fc1(x_c5)
        x_fc1 = F.gelu(x_fc1)
        
        x_out = self.fc2(x_fc1)

        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)



class Discriminator(nn.Module):
    def __init__(self, in_d_co_domain, d_co_domain, modes_schedule, kernel_dim=16, pad = 0, factor = 3/4):
        super(Discriminator, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.in_d_co_domain = in_d_co_domain # input channel
        self.d_co_domain = d_co_domain 
        self.factor = factor
        self.padding = pad  # pad the domain if input is non-periodic
        self.kernel_dim = kernel_dim
        self.fc0 = nn.Conv2d(self.in_d_co_domain, self.d_co_domain, 1)
        modes_schedule = modes_schedule+modes_schedule[-1::-1]
        self.conv0 = SpectralConv2dUno(self.d_co_domain, 2*factor*self.d_co_domain, modes_schedule[0], modes_schedule[0])

        self.conv1 = SpectralConv2dUno(2*factor*self.d_co_domain, 4*factor*self.d_co_domain, modes_schedule[1], modes_schedule[1])

        self.conv2 = SpectralConv2dUno(4*factor*self.d_co_domain, 8*factor*self.d_co_domain, modes_schedule[2], modes_schedule[2])
        
        self.conv2_1 = SpectralConv2dUno(8*factor*self.d_co_domain, 16*factor*self.d_co_domain, modes_schedule[3], modes_schedule[3])
        
        self.conv2_9 = SpectralConv2dUno(16*factor*self.d_co_domain, 8*factor*self.d_co_domain, modes_schedule[4], modes_schedule[4])
        

        self.conv3 = SpectralConv2dUno(16*factor*self.d_co_domain, 4*factor*self.d_co_domain, modes_schedule[5], modes_schedule[5])

        self.conv4 = SpectralConv2dUno(8*factor*self.d_co_domain, 2*factor*self.d_co_domain, modes_schedule[6], modes_schedule[6])

        self.conv5 = SpectralConv2dUno(4*factor*self.d_co_domain, self.d_co_domain, modes_schedule[7], modes_schedule[7]) # will be reshaped

        self.w0 = pointwise_op(self.d_co_domain,2*factor*self.d_co_domain) #
        
        self.w1 = pointwise_op(2*factor*self.d_co_domain, 4*factor*self.d_co_domain) #
        
        self.w2 = pointwise_op(4*factor*self.d_co_domain, 8*factor*self.d_co_domain) #
        
        self.w2_1 = pointwise_op(8*factor*self.d_co_domain, 16*factor*self.d_co_domain)
        
        self.w2_9 = pointwise_op(16*factor*self.d_co_domain, 8*factor*self.d_co_domain)
        
        self.w3 = pointwise_op(16*factor*self.d_co_domain, 4*factor*self.d_co_domain) #
        
        self.w4 = pointwise_op(8*factor*self.d_co_domain, 2*factor*self.d_co_domain)
        
        self.w5 = pointwise_op(4*factor*self.d_co_domain, self.d_co_domain) # will be reshaped

        self.fc1 = nn.Conv2d(2*self.d_co_domain, 4*self.d_co_domain, 1)
        self.fc2 = nn.Conv2d(4*self.d_co_domain, in_d_co_domain-2, 1)
        
        # kernel for last functional operation

        self.knet = self._kernel(2, self.kernel_dim)

    def _kernel(self, in_chan=2, up_dim=32):
        """
            Kernel network apply on grid
        """
        layers = nn.Sequential(
                    nn.Conv2d(in_chan, up_dim, 1, bias=True), torch.nn.GELU(),
                    nn.Conv2d(up_dim, up_dim, 1, bias=True), torch.nn.GELU(),
                    nn.Conv2d(up_dim, 1, 1, bias=False)
                )
        return layers
    def forward(self, x):
        batch_size = x.shape[0]
        res1 = x.shape[2]
        res2 = x.shape[3]
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
         
        x_fc0 = self.fc0(x)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = F.pad(x_fc0, [0,self.padding, 0,self.padding])
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]
        x1_c0 = self.conv0(x_fc0,int(D1*self.factor),int(D2*self.factor))
        x2_c0 = self.w0(x_fc0,int(D1*self.factor),int(D2*self.factor))
        x_c0 = x1_c0 + x2_c0
        x_c0 = F.gelu(x_c0)
        #print(x.shape)

        x1_c1 = self.conv1(x_c0 ,D1//2,D2//2)
        x2_c1 = self.w1(x_c0 ,D1//2,D2//2)
        x_c1 = x1_c1 + x2_c1
        x_c1 = F.gelu(x_c1)
        #print(x.shape)

        x1_c2 = self.conv2(x_c1 ,D1//4,D2//4)
        x2_c2 = self.w2(x_c1 ,D1//4,D2//4)
        x_c2 = x1_c2 + x2_c2
        x_c2 = F.gelu(x_c2 )
        #print(x.shape)
        
        x1_c2_1 = self.conv2_1(x_c2,D1//8,D2//8)
        x2_c2_1 = self.w2_1(x_c2,D1//8,D2//8)
        x_c2_1 = x1_c2_1 + x2_c2_1
        x_c2_1 = F.gelu(x_c2_1)
        
        x1_c2_9 = self.conv2_9(x_c2_1,D1//4,D2//4)
        x2_c2_9 = self.w2_9(x_c2_1,D1//4,D2//4)
        x_c2_9 = x1_c2_9 + x2_c2_9
        x_c2_9 = F.gelu(x_c2_9)
        x_c2_9 = torch.cat([x_c2_9, x_c2], dim=1) 

        x1_c3 = self.conv3(x_c2_9,D1//2,D2//2)
        x2_c3 = self.w3(x_c2_9,D1//2,D2//2)
        x_c3 = x1_c3 + x2_c3
        x_c3 = F.gelu(x_c3)
        x_c3 = torch.cat([x_c3, x_c1], dim=1)

        x1_c4 = self.conv4(x_c3,int(D1*self.factor),int(D2*self.factor))
        x2_c4 = self.w4(x_c3,int(D1*self.factor),int(D2*self.factor))
        x_c4 = x1_c4 + x2_c4
        x_c4 = F.gelu(x_c4)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)

        x1_c5 = self.conv5(x_c4,D1,D2)
        x2_c5 = self.w5(x_c4,D1,D2)
        x_c5 = x1_c5 + x2_c5
        x_c5 = F.gelu(x_c5)
        

        x_c5 = torch.cat([x_c5, x_fc0], dim=1)
        if self.padding!=0:
            x_c5 = x_c5[..., :-self.padding, :-self.padding]

        x = x_c5

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        kx = self.knet(grid)
        kx = kx.view(batch_size,-1, 1)
        x = x.view(batch_size,-1, 1)
        x = torch.einsum('bik,bik->bk', kx, x)/(res1*res2)

        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)