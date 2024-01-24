import yaml
import torch.nn as nn
import torchvision.utils as utils
from models import *
from utils.load import *
from math import log2, exp, log
from tqdm import tqdm
import os
import torch
from utils.sample_grid import sample_grid
from utils.load import load_model
import torch
import matplotlib.pyplot as plt

class ConfigManager:
    def __init__(self, filepath):
        self.filepath = filepath
        self.config = None
        with open(self.filepath, 'r') as file:
            self.config = yaml.safe_load(file)
        self.dataset = None
        self.dataset_choices = {'mnist':load_mnist, 'cifar10':load_cifar, 'neurop_32':None}
    def get_model(self):
    # try:
        constructor = eval(self.config['model']['name'])
        # except:
        #     raise Exception('Invalid model choice')
        model = constructor(**self.config['model']['details'][self.config['model']['name']])
        if self.config['train']['load_path'] is not None:
            model = load_model(model, self.config['train']['load_path'])
        return model
    def get_data(self):
        assert self.config['data']['dataset'] in self.dataset_choices, "Invalid dataset choice"
        dataset = self.dataset_choices[self.config['data']['dataset']]
        return dataset
    
class Trainer:
    def __init__(self, config_manager) -> None:
        self.cm = config_manager
        self.is_gan = self.cm.config['model']['is_gan']
    def train(self):

        def print_reserved_memory():
            reserved_memory = torch.cuda.memory_reserved()
            print(f"Reserved memory: {reserved_memory} bytes")

        print_reserved_memory()
        self.model = self.cm.get_model().to('cuda')
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters: {total_params}")
        epochs = self.cm.config['train']['epochs']
        lr = self.cm.config['train']['lr']
        save_freq = self.cm.config['train']['save_freq']
        test_freq = self.cm.config['train']['test_freq']
        save_path = self.cm.config['train']['save_path']
        test_path = self.cm.config['train']['test_path']
        train_set, test_set = self.load_dataset()
        test_lowest = self.cm.config['train']['test_lowest_resolution']
        epochs = tqdm(range(epochs), position=0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if self.is_gan:
            loss_curve_G = []
            loss_curve_D = []   
        else:
            loss_curve = []
        for i in epochs:
            if self.is_gan:
                avg_loss_G = 0
                avg_loss_D = 0
                loss_G = 0
                loss_D = 0
            else:
                avg_loss = 0
            batches = tqdm(train_set, position=1, leave=False)
            for k, j in enumerate(batches):
                j = j[0].to('cuda')
                if not self.is_gan:
                    loss = self.model(j)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    nom_loss, unit = self.format_loss(loss, j)
                    avg_loss += nom_loss/len(batches)
                    batches.set_description(f"Epoch {i}, Batch {k}, Loss: {nom_loss} {unit}")
                else:
                    lG, lD = self.model(j)
                    optimizer.zero_grad()
                    if lG is None:
                        lD.backward()
                        loss_D = lD.item()
                        lG = torch.tensor(0.0)
                    else:
                        lG.backward()
                        loss_G = lG.item()
                        lD = torch.tensor(0.0)
                    optimizer.step()
                    avg_loss_D += lD.item()/(len(batches)//self.model.n_critic)
                    avg_loss_G += lG.item()/(len(batches)-len(batches)//self.model.n_critic)
                    batches.set_description(f"Epoch {i}, Batch {k}, Loss: G {loss_G}, D {loss_D}")
            if self.is_gan:
                epochs.set_description(f"Epoch {i}, Avg_loss: G {avg_loss_G}, D {avg_loss_D}; Last_batch_loss: G {loss_G}, D {loss_D}")
                loss_curve_G.append(avg_loss_G)
                loss_curve_D.append(avg_loss_D)
            else:
                epochs.set_description(f"Epoch {i}, Avg_loss: {avg_loss} {unit}, Last_batch_loss: {nom_loss} {unit}")
                loss_curve.append(avg_loss)
            if i % save_freq == 0:
                if self.cm.config['train']['save_name'] == None:
                    torch.save(self.model.state_dict(), save_path+f"{self.cm.config['model']['name']}_epoch_{i}.pth")
                else:
                    torch.save(self.model.state_dict(), save_path+f"{self.cm.config['train']['save_name']}_epoch_{i}.pth")
            if i % test_freq == 0:
                sample_grid(self.model, test_lowest, test_path+f"{self.cm.config['model']['name']}_epoch_{i}.png", clip_range=(0, 255))
        if self.is_gan:
            plt.clf()
            plt.plot(loss_curve_G, label="Generator")
            plt.plot(loss_curve_D, label="Discriminator")
            plt.yscale('symlog')
            plt.legend()
            plt.savefig(save_path+f"{self.cm.config['model']['name']}_loss_curve.png")
        else:
            plt.clf()  # Clear the figure
            plt.plot(loss_curve)
            plt.yscale('symlog')
            plt.savefig(save_path+f"{self.cm.config['model']['name']}_loss_curve.png")
            
    def format_loss(self, loss, j):
        if self.cm.config['model']['loss_unit'] == "bpd":
            nom_loss = loss.item()*log2(exp(1)) / (np.prod(j.shape[-2:]) * j.shape[0])
            unit = "bpd"
        else:
            nom_loss = loss.item() / (np.prod(j.shape[-2:]) * j.shape[0])
            unit = ""
        return nom_loss, unit

    def load_dataset(self):
        path = self.cm.config['data']['path'] + self.cm.config['data']['dataset']
        args = self.cm.config['data'].copy()
        args['path'] = path
        del args['dataset']
        train_set, test_set = self.cm.get_data()(**args)
        return train_set, test_set
