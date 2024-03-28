from tools import ConfigManager, Trainer
import torch
if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    cm = ConfigManager('config.yaml')
    trainer = Trainer(cm)
    trainer.train()