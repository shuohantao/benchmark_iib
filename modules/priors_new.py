import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np
from numpy.linalg import svd
from .fno import *
import warnings


class Prior(nn.Module):
    def __init__(self):
        super(Prior, self).__init__()

        
    def sample(self, shape_vec):
        return NotImplementedError()
    
    def scale_factor(self, shape_vec, target_energy=0.2, samples=100):
        '''
        Computes scale parameter to ensure that draws have L^2 norm equal to `target_energy`.

        For MNIST, the L^2 energy of the training dataset is on average about ~0.4. 
        It seems sensible to target that for the prior, too.
        '''
        l2norm = 0
        for sample in range(samples):
            z = self.sample(shape_vec)
            l2norm += z.square().mean().cpu().detach().numpy()
        
        l2norm /= samples
        return l2norm**(-0.5) * target_energy

class StandardNormal(Prior):

    def __init__(self):
        super(StandardNormal, self).__init__()

    def __repr__(self):
        return "StandardNormal"

    #standard noise
    def sample(self,shape_vec):
        x = torch.randn(shape_vec[0],1,shape_vec[2],shape_vec[3]).to(device)
        return self.Qmv(x)

    #covariance
    def Qmv(self,v):
        return v

    #loss structure
    #g*a means the Q in reverse drift is learned implicitly in a
    #self.Qmv(g*a) means the Q is given explictly
    def Q_g2_s(self, g,a): 
        return g*a 

class FNOprior(Prior):
    def __init__(self,k1=14,k2=14, scale=1):
        super(FNOprior, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.scale = scale
        self.conv = SpectralConv2d(1,1,k1,k2, rand = False).to(device)

    def __repr__(self):
        return "FNOprior(k1=%d, k2=%d)" %(self.k1,self.k2)

    def sample(self,shape_vec):
        # Generate white-noise latent
        xhat = torch.randn(shape_vec[0],1,shape_vec[2],shape_vec[3]).to(device)
        x = torch.fft.ifft2(xhat, norm='forward').real
        return self.Qmv(x) * self.scale

    def Qmv(self,v):
        return self.conv(v)

    def Q_g2_s(self, g,a): 
        return g*a #Qmv(g*a)
    



class ImplicitConv(Prior):
    """
    Compute the covariance matrix as

    Q = conv(K)^{-1/2}

    where K is a  stencil, e.g., the standard 5-point Laplacian.
    """

    def __init__(self,K=None, scale=1, smoothness=2):
        super(ImplicitConv, self).__init__()
        if smoothness <= 1:
            warnings.warn("Smoothness of Laplacian prior must be strictly greater than 1 to be well defined.")
        
        self.K = K
        self.scale = scale
        self.smoothness = smoothness

    def __repr__(self):
        Knum = self.K.numpy()
        Kstr = ';'.join(','.join(map(lambda x: str(int(x)), row)) for row in Knum)
        return "ImplicitConv(K=%s)" %(Kstr)

    def sample(self,shape_vec):
        xhat = torch.randn(shape_vec[0],1,shape_vec[2],shape_vec[3]).to(device)
        x = torch.fft.ifft2(xhat, norm='forward').real
        return self.Qmv(x) * self.scale

    def Qmv(self,v):
        return self.compConv(v, fun = lambda x: x**(-self.smoothness/2))[0]

    def Q_g2_s(self, g,a): #method
        return g*a #self.Qmv(g*a) 



    def compConv(self,x,fun= lambda y : y, norm='forward'):
        """
        compute fun(conv_op(K))*x assuming periodic boundary conditions on x

        where
        x       - are images, torch.tensor, shape=N x 1 x nx x ny
        fun     - is a function applied to the operator (as a matrix-function, not
    component-wise), default fun(x)=x
        """

        n = x.shape
        K = self.K
        m = K.shape
        mid1 = (m[0]-1)//2
        mid2 = (m[1]-1)//2
        Bp = torch.zeros(n[2],n[3], device = device)
        Bp[0:mid1+1,0:mid2+1] = K[mid1:,mid2:]
        Bp[-mid1:, 0:mid2 + 1] = K[0:mid1, -(mid2 + 1):]
        Bp[0:mid1 + 1, -mid2:] = K[-(mid1 + 1):, 0:mid2]
        Bp[-mid1:, -mid2:] = K[0:mid1, 0:mid2]
        xh = torch.fft.rfft2(x, norm=norm)
        Bh = torch.fft.rfft2(Bp, norm='backward')
        lam = fun(torch.abs(Bh)).to(device)
        xh = xh.to(device)
        lam[0, 0] = 0.0
        xBh = xh * lam.unsqueeze(0).unsqueeze(0)
        xB = torch.fft.irfft2(xBh, norm=norm)
        return xB,lam


class CombinedConv(Prior):
    """
    Compute the covariance matrix as

    Q = conv(K)^{-1/2}

    where K is a  stencil, e.g., the standard 5-point Laplacian.

    `scale_ratio` denotes the ratio of FNO prior scale to Laplacian prior scale
    """

    def __init__(self,K=None,k1=14,k2=14,scale=1,scale_ratio=None, smoothness=2):
        super(CombinedConv, self).__init__()
        self.scale = scale
        if scale_ratio is not None:
            scale_lap = 1
            scale_fno = scale_ratio * scale_lap
            self.fno_prior = FNOprior(k1=k1, k2=k2, scale=scale_fno)
            self.lap_prior = ImplicitConv(K, smoothness=smoothness, scale=scale_lap)
        else:
            self.fno_prior = FNOprior(k1=k1, k2=k2)
            self.lap_prior = ImplicitConv(K, smoothness=smoothness)

    def sample(self,shape_vec):
        return (1/2**(0.5)) * (self.fno_prior.sample(shape_vec) + self.lap_prior.sample(shape_vec)) * self.scale

if __name__ == '__main__':
    K = torch.zeros(3,3)
    K[1,1] = 4.0
    K[0, 1] = -1
    K[1, 0] = -1
    K[2, ] = -1

    x = torch.randn(10,1,16,32)

    y = torch.nn.functional.conv2d(x,K.unsqueeze(0).unsqueeze(0),padding=1)

    convOp = ImplicitConv(K)
    yt = convOp.compConv(x)[0]

    rel_err = torch.norm((y-yt)[:,:,1:-2,1:-2])/torch.norm(y[:,:,1:-2,1:-2])
    print(rel_err)





