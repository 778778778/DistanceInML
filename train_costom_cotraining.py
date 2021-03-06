import warnings
from typing import List
import yaml
from generalframework.dataset.ACDC_helper import get_costom_split_dataloders,get_ACDC_split_dataloders
from generalframework.loss import get_loss_fn
from generalframework.models import Segmentator
from generalframework.trainer import CoTrainer
from generalframework.utils import yaml_parser, dict_merge, fix_all_seed
import torch
from torchvision.transforms import ToPILImage

warnings.filterwarnings('ignore')
parser_args = yaml_parser()
def get_models(config):
    num_models = config['Lab_Partitions']['num_models']
    model = []
    model.append(Segmentator(arch_dict=config['Arch'],optim_dict=config['Optim'], scheduler_dict=config['Scheduler']))
    model.append(Segmentator(arch_dict=config['Brch'],optim_dict=config['Optim'], scheduler_dict=config['Scheduler']))
    return model
    # for i in range(num_models):
    #     return [Segmentator(arch_dict=config['Brch'], optim_dict=config['Optim'], scheduler_dict=config['Scheduler'])
    #             for _ in range(num_models)]
if __name__ == '__main__':

    with open('config/my_config.yaml', 'r') as f:
        config = yaml.load(f.read())
    config = dict_merge(config, parser_args, True)
    fix_all_seed(int(config['Seed']))

    labeled_dataloaders, unlab_dataloader, val_dataloader = get_ACDC_split_dataloders(config)

    segmentators: List[Segmentator] = get_models(config)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        criterions = {'sup': get_loss_fn('cross_entropy'),
                      'jsd': get_loss_fn('jsd'),
                      'adv': get_loss_fn('jsd')}

    cotrainner = CoTrainer(segmentators=segmentators,
                           labeled_dataloaders=labeled_dataloaders,
                           unlabeled_dataloader=unlab_dataloader,
                           val_dataloader=val_dataloader,
                           criterions=criterions,
                           adv_scheduler_dict=config['Adv_Scheduler'],
                           cot_scheduler_dict=config['Cot_Scheduler'],
                           adv_training_dict=config['Adv_Training'],
                           **config['Trainer'],
                           whole_config=config)


    cotrainner.start_training(**config['StartTraining'])