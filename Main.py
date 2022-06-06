import numpy as np
import torch

loss = torch.nn.KLDivLoss()

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "pascal_voc": VOCDataSet
    }[name]
if __name__ == '__main__':
    x = torch.rand(5,2)
    y = torch.rand(5,2)




