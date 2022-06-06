import random
import re
from copy import deepcopy as dcopy
from itertools import repeat
from pathlib import Path
from typing import Callable, Dict, List, Match, Pattern, TypeVar, Iterable
from torch.utils import data
import numpy as np
import torch
from torch.utils.data import DataLoader,Sampler

from . import MedicalImageDataset

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", torch.Tensor, np.ndarray)


def id_(x):
    return x


def map_(fn: Callable[[A], B], iter_: Iterable[A]) -> List[B]:
    return list(map(fn, iter_))


class PatientSampler(Sampler):
    def __init__(self, dataset: MedicalImageDataset, grp_regex, shuffle=False, quite=False) -> None:
        filenames: List[str] = dataset.filenames[dataset.subfolders[0]]
        # Might be needed in case of escape sequence fuckups
        # self.grp_regex = bytes(grp_regex, "utf-8").decode('unicode_escape')
        self.grp_regex = grp_regex
        # Configure the shuffling function
        self.shuffle: bool = shuffle
        self.shuffle_fn: Callable = (lambda x: random.sample(x, len(x))) if self.shuffle else id_
        if not quite:
            print(f"Grouping using {self.grp_regex} regex")
        # assert grp_regex == "(patient\d+_\d+)_\d+"
        # grouping_regex: Pattern = re.compile("grp_regex")
        grouping_regex: Pattern = re.compile(self.grp_regex)

        stems: List[str] = [Path(filename).stem for filename in filenames]  # avoid matching the extension
        matches: List[Match] = map_(grouping_regex.match, stems)
        patients: List[str] = [match.group(1) for match in matches]

        unique_patients: List[str] = list(set(patients))
        assert len(unique_patients) < len(filenames)
        if not quite:
            print(f"Found {len(unique_patients)} unique patients out of {len(filenames)} images")

        self.idx_map: Dict[str, List[int]] = dict(zip(unique_patients, repeat(None)))
        for i, patient in enumerate(patients):
            if not self.idx_map[patient]:
                self.idx_map[patient] = []

            self.idx_map[patient] += [i]
        assert sum(len(self.idx_map[k]) for k in unique_patients) == len(filenames)
        if not quite:
            print("Patient to slices mapping done")

    def __len__(self):
        return len(self.idx_map.keys())

    def __iter__(self):
        values = list(self.idx_map.values())
        shuffled = self.shuffle_fn(values)
        return iter(shuffled)


def get_ACDC_dataset(dataset_dict: dict, dataloader_dict: dict, quite=False, mode1='train', mode2='val'):
    print(dataset_dict)
    train_set = MedicalImageDataset(mode=mode1, quite=quite, **dataset_dict)
    val_set = MedicalImageDataset(mode=mode2, quite=quite, **dataset_dict)

    # train_loader = DataLoader(train_set, **{**dataloader_dict, **{'batch_sampler': None}})  #包括 imgs，metainfo，filenames，imgs包括 img 和gt img[0]s是标签 int
    #
    #
    #
    # if dataloader_dict.get('batch_sampler') is not None:
    #     val_sampler = eval(dataloader_dict.get('batch_sampler')[0])(dataset=val_set, **dataloader_dict.get('batch_sampler')[1])
    #     val_loader = DataLoader(val_set, batch_sampler=val_sampler, batch_size=1)
    # else:
    #     val_loader = DataLoader(val_set, **{**dataloader_dict, **{'shuffle': False, 'batch_size': 1}})
    return {'train': train_set, 'val': val_set}

def get_ACDC_split_dataloders(config):

    dataset = get_ACDC_dataset(config['Dataset'], config['Lab_Dataloader'])
    train_set = dataset['train']
    val_set = dataset['val']
    partition_ratio = 0.2
    train_ids = np.arange(len(train_set)) #获取总数索引
    np.random.shuffle(train_ids)
    partial_size = int(partition_ratio*len(train_set))
    train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
    train_remain_sampler = data.sampler.SubsetRandomSampler(train_ids[partial_size:])
    labeled_dataloaders = []
    labeled_dataloaders.append(DataLoader(train_set,batch_size=1,sampler=train_sampler,num_workers=0,pin_memory=False))
    labeled_dataloaders.append(
        DataLoader(train_set, batch_size=1, sampler=train_sampler, num_workers=0, pin_memory=True))

    unlab_dataloader = DataLoader(train_set,batch_size=1,sampler=train_remain_sampler,num_workers=0,pin_memory=False)

    val_dataloader = DataLoader(val_set,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)


    return labeled_dataloaders, unlab_dataloader, val_dataloader
def extract_patients(dataloader: DataLoader, patient_ids: List[str]):
    '''
     extract patients from ACDC dataset provding patient_ids
    :param dataloader:
    :param patient_ids:
    :return:
    '''
    assert isinstance(patient_ids, list)
    bpattern = lambda d: 'patient%.3d' % int(d)
    patterns = re.compile('|'.join([bpattern(id) for id in patient_ids]))
    files: Dict[str, List[str]] = dcopy(dataloader.dataset.imgs)
    files = {k: [s for s in file if re.search(patterns, s)] for k, file in files.items()}
    for v in files.values():
        v.sort()
    new_dataloader = dcopy(dataloader)
    new_dataloader.dataset.imgs = files
    new_dataloader.dataset.filenames = files
    return new_dataloader
def get_costom_split_dataloders(config):

    dataloders = get_costom_split_dataloders(config['Dataset'], config['Lab_Dataloader'])
    partition_ratio = config['Lab_Partitions']['partition_sets'] # 划分比例
    train_dataset_size = len(dataloders['train']) #获取总数
    print('Training on number of samples:', train_dataset_size)

    train_ids = np.arange(train_dataset_size) #获取总数索引
    np.random.shuffle(train_ids)
    partial_size = int(partition_ratio*train_dataset_size)
    train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
    train_remain_sampler = data.sampler.SubsetRandomSampler(train_ids[partial_size:])

    labeled_dataloaders = []






    return labeled_dataloaders


# def extract_patients(dataloader: DataLoader, patient_ids: List[str]):
#     '''
#      extract patients from ACDC data provding patient_ids
#     :param dataloader:
#     :param patient_ids:
#     :return:
#     '''
#     assert isinstance(patient_ids, list)
#     # bpattern = lambda d: 'patient%.3d' % int(d)
#     # patterns = re.compile('|'.join([bpattern(id) for id in patient_ids]))
#     files: Dict[str, List[str]] = dcopy(dataloader.dataset.imgs)
#
#
#
#     return

# todo new data interface for data access.
# highlight the sampler=Subsetrandomsampler is mutaully exclusive with batch_size, shuffle, sampler, and drop_last
# and the best way to implement the batch_sampler is to change directed the stored images.
#
# @export
# def get_ACDC_datasets(dataset_dict: dict, mode1='train', mode2='val', quite=False)-> Tuple[MedicalImageDataset, MedicalImageDataset]:
#     dataset_dict['root_dir'] = Path(__file__).parents[2] / 'data' / 'ACDC-all'
#     train_set = MedicalImageDataset(mode=mode1, quite=quite, **dataset_dict)
#     val_set = MedicalImageDataset(mode=mode2, quite=quite, **dataset_dict)
#     return train_set, val_set
#
# def get_split_idex(train_dataset):
#
#
# def extract_ids(data: MedicalImageDataset, patterns: List[str]):
#     pass

# def get_ACDC_dataloaders():
#     pass

