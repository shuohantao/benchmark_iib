import yaml
import torch.nn as nn
import torchvision.utils as utils
from models.CouplingFlowAR import CouplingFlowAR
import torchvision.transforms.functional as F
from models.MultilevelDiff import MultilevelDiff
from utils.load import *
from math import log2, exp, log
from tqdm import tqdm
import os
import torch
from utils.sample_grid import sample_grid
import matplotlib.pyplot as plt

class ConfigManager:
    def __init__(self, filepath):
        self.filepath = filepath
        self.config = None
        with open(self.filepath, 'r') as file:
            self.config = yaml.safe_load(file)
        self.model_choices = {'coupling_ar':CouplingFlowAR, 'multilevelDiff':MultilevelDiff}
        self.dataset = None
        self.dataset_choices = {'mnist':load_mnist, 'cifar10':load_cifar, 'neurop_32':None}
    def get_model(self):
        try:
            constructor = self.model_choices[self.config['model']['name']]
        except:
            raise Exception('Invalid model choice')
        model = constructor(**self.config['model']['details'][self.config['model']['name']])
        return model
    def get_data(self):
        assert self.config['data']['dataset'] in self.dataset_choices, "Invalid dataset choice"
        dataset = self.dataset_choices[self.config['data']['dataset']]
        return dataset
    
class Trainer:
    def __init__(self, config_manager) -> None:
        self.cm = config_manager
        self.model = config_manager.get_model().to('cuda')
    def train(self):
        print(type(self.model))
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
        loss_curve = []
        for i in epochs:
            avg_loss = 0
            batches = tqdm(train_set, position=1, leave=False)
            for k, j in enumerate(batches):
                j = j[0].to('cuda')
                loss = self.model(j)
                print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()/len(batches)*log2(exp(1)) / (np.prod(j.shape[-2:]) * len(j))
                batches.set_description(f"Epoch {i}, Batch {k}, Loss: {loss.item()*log2(exp(1)) / (np.prod(j.shape[-2:]) * len(j))}")
            epochs.set_description(f"Epoch {i}, Avg_loss: {avg_loss}, Last_batch_loss: {loss.item()*log2(exp(1)) / (np.prod(j.shape[-2:]) * len(j))}")
            loss_curve.append(avg_loss)
            if i % save_freq == 0:
                torch.save(self.model.state_dict(), save_path+f"epoch_{i}.pth")
            if i % test_freq == 0:
                sample_grid(self.model, test_lowest, test_path+f"epoch_{i}.png")
        plt.plot(loss_curve)
        plt.savefig(save_path+f"{self.cm.config['model']['name']}loss_curve.png")
        
    def load_dataset(self):
        path = self.cm.config['data']['path'] + self.cm.config['data']['dataset']
        args = self.cm.config['data'].copy()
        args['path'] = path
        del args['dataset']
        train_set, test_set = self.cm.get_data()(**args)
        return train_set, test_set