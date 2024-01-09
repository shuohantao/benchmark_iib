from tools import ConfigManager, Trainer
if __name__ == '__main__':
    cm = ConfigManager('config.yaml')
    trainer = Trainer(cm)
    trainer.train()