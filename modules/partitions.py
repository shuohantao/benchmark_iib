import torch
def checkerBoard(shape, even):
    h, w = shape[-2:]
    x, y = torch.arange(h, dtype=torch.int32), torch.arange(w, dtype=torch.int32)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    mask = torch.fmod(xx + yy, 2)
    mask = mask.to(torch.float32).view(1, 1, h, w)
    if even:
        mask = 1 - mask
    return mask
def checkerBoard1D(shape, even):
    w = shape
    x = torch.arange(w, dtype=torch.int32)
    mask = torch.fmod(x, 2)
    mask = mask.to(torch.float32).view(1, w)
    if even:
        mask = 1 - mask
    return mask
def halfMask(shape, even):
    h, w = shape[-2:]
    mask = torch.zeros(1, 1, h, w)
    mask[:, :, :h//2, :] = 1
    if even:
        mask = 1 - mask
    return mask

def channelMask(shape, even):
    c_in = shape[1]
    mask = torch.cat([torch.ones(c_in//2, dtype=torch.float32),
                      torch.zeros(c_in-c_in//2, dtype=torch.float32)])
    mask = mask.view(1, c_in, 1, 1)
    if even:
        mask = 1 - mask
    return mask

def patchMask(h, w, ph, pw, py, px):
    mask = torch.ones(1, 1, h, w)
    mask[:, :, py:py+ph, px:px+pw] = 0
    return mask

def freqInpaint(masked_modes, inpaint_modes=1):
    mask = torch.zeros(1, 1, masked_modes+inpaint_modes, masked_modes+inpaint_modes) + torch.ones(1, 1, masked_modes, masked_modes)
    return mask