import torch
import torch.nn as nn
import torch.nn.functional as F
from haar_pytorch import HaarForward, HaarInverse

def low_pass(x, modes, device):
    x = x.to(device)
    B, C, H, W = x.shape
    if modes == W:
        F = _perform_dft(x)
        F_low_modes = F
        x_rec = _perform_idft(F_low_modes)
    else:
        N = W
        n_rem = (N - modes) // 2
        F = _perform_dft(x)
        F_low_modes = F[:, :, n_rem:-n_rem, n_rem:-n_rem]
        x_rec = _perform_idft(F_low_modes)
    return x_rec

def frequency_seg(x, modes, step, device):
    x_low = low_pass(x, modes, device)
    x = x.to(device)
    B, C, H, W = x.shape
    N = W
    n_del = (N - modes) // 2
    F_high_modes = _perform_dft(x)
    F_high_modes[:, :, n_del:-n_del, n_del:-n_del] = 0
    seg_list=[]
    con = _perform_dft(x_low)
    con = F.pad(con, (step, step, step, step), 'constant', 0)
    for i in range(n_del//step - 1):
        i += 1
        f = F_high_modes[:, :, n_del - i*step:- (n_del - i*step), n_del - i*step:- (n_del - i*step)]
        freq_seg_img = _perform_idft(f)
        seg_list.append([freq_seg_img, _perform_idft(con)])
        con = F.pad(f+con, (step, step, step, step), 'constant', 0)
    f = F_high_modes
    freq_seg_img = _perform_idft(f)
    seg_list.append([freq_seg_img, _perform_idft(con)])
    return x_low, seg_list
def pad_zeros(x, width):
    torch.cat()
def _perform_dft(img):
    F = torch.fft.fft2(img, norm='ortho')
    F = torch.fft.fftshift(F)
    return F

def _perform_idft(F):
    F = torch.fft.ifftshift(F)
    img = torch.fft.ifft2(F, norm='ortho').float()
    return img

def wavelet_seg(x, num_scales):
    haar = HaarForward()
    segments = []
    for i in range(num_scales):
        transformed = haar(x)
        undersampled = transformed[:, 0, ...].unsqueeze(1).to(x.device)
        coeffs = transformed[:, 1:, ...].to(x.device)
        segments.append([undersampled, coeffs])
        x = undersampled
    return segments