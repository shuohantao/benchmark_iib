import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import ConfigManager
from utils.metrics import *
from utils.load import load_mnist, load_darcy
from utils.load import load_model
from modules.act_norm import ActNorm
import pickle
import os
import torch.nn.functional as F
from tqdm import tqdm
with torch.no_grad():
    cm = ConfigManager('config.yaml')
    resolutions = cm.config['MMD_test']['resolutions']
    num_batch = cm.config['MMD_test']['num_batch']
    model = cm.get_model().to('cuda')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    train, test = load_darcy(batch_size=cm.config['MMD_test']['batch_size'], shape_setting=[[max(cm.config["MMD_test"]["resolutions"]), 1]])
    val_samp = next(iter(test))[0]
    versions = []
    for resolution in resolutions[:-1]:
        downsampled = F.interpolate(val_samp, size=(resolution, resolution), mode='bilinear', align_corners=False)
        versions.append(downsampled)
    versions.append(val_samp)
    versions = [i.view(-1, np.prod(i.shape[-2:])).float() for i in versions]
    mmds = [0]*len(resolutions)
    for j in range(num_batch):
        test_samples = next(iter(test))[0].to('cuda')
        for i, r in tqdm(enumerate(resolutions)):
            X_samples = F.interpolate(test_samples, size=r, mode='bilinear')
            #X_samples = model.sample(X_samples, num_context=16, autoregressive=False, num_samples=cm.config['MMD_test']['batch_size'], resolution=r, device="cuda").cpu()
            X_samples = model.sample(num_samples=cm.config['MMD_test']['batch_size'], resolution=r, device="cuda")
            X_samples = X_samples.view(-1, np.prod(X_samples.shape[-2:])).float().cpu()
            mmds[i] += mmd(versions[i], X_samples).detach().cpu().numpy()/num_batch
            print(f"resolution: {r}; MMD: {mmds[i]}")
            del X_samples
            torch.cuda.empty_cache()
    mmds = [mmds, resolutions]
    with open(os.path.join(cm.config["MMD_test"]["pickle_save_dir"], f'{cm.config["train"]["load_path"].split("/")[-1].split(".")[0]}_mmds_{total_params}.pkl'), "wb") as f:
        pickle.dump(mmds, f)
    for i, r in enumerate(mmds[1]):
        print(f"resolution: {r}; MMD: {mmds[0][i]}")