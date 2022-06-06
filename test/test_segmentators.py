from generalframework.models import Segmentator
import yaml, os
import torch
import torch.nn as nn
with open('../config/ACDC_config.yaml','r') as f:
    config = yaml.load(f.read())

print(config)

model = Segmentator(arch_dict=config['Arch'],optim_dict=config['Optim'],scheduler_dict=config['Scheduler'])
img = torch.randn(1,1,512,512)
model.predict(img)
target = torch.randint(0,2,(1,1,512,512))
model.update(img,target,nn.CrossEntropyLoss())

