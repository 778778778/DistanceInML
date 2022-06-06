from generalframework.models import Segmentator
from generalframework.trainer import Trainer
from generalframework.dataset.ACDC_helper import get_ACDC_split_dataloders

from generalframework.loss import get_loss_fn
import yaml, os
import torch
import torch.nn as nn
import warnings

with open('../config/my_config.yaml', 'r') as f:
    config = yaml.load(f.read())

print(config)

dataloders = get_ACDC_split_dataloders(config['Dataset'], config['Dataloader'])

model = Segmentator(arch_dict=config['Arch'], optim_dict=config['Optim'], scheduler_dict=config['Scheduler'])

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    criterion = get_loss_fn(config['Loss'].get('name'), **{k: v for k, v in config['Loss'].items() if k != 'name'})

trainer = Trainer(model, dataloaders=dataloders, criterion=criterion, **config['Trainer'])
trainer.start_training()
