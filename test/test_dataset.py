from generalframework.dataset import MedicalImageDataset, PILaugment, segment_transform, CityscapesDataset
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader
from generalframework.utils import iterator_
from pathlib import Path
from generalframework.dataset.augment import Scale, RandomRotate, Compose
import numpy as np


def test_dataset():
    dataroot: str = r'E:/PycharmProject/FCN/Datasets/ACDC-all'
    train_set = MedicalImageDataset(dataroot, 'train', subfolders=['img', 'gtt'],
                                    transform=segment_transform((256, 256)),
                                    augment=PILaugment, equalize=None, pin_memory=True
                                    )
    train_loader = DataLoader(train_set, batch_size=10, num_workers=1)

    n_batches = len(train_loader)

    for i, (imgs, metainfo, filenames) in enumerate(train_loader):
        print(imgs)
        print(metainfo)
        print(filenames)
        time.sleep(2)


def test_iter():
    dataroot: str = r'E:/PycharmProject/FCN/Datasets/ACDC-all'
    train_set = MedicalImageDataset(dataroot, 'train', subfolders=['img', 'gt'],
                                    transform=segment_transform((256, 256)),
                                    augment=PILaugment, equalize=None, pin_memory=False,

                                    )
    train_loader_1 = DataLoader(train_set, batch_size=10, num_workers=1)
    train_loader_2 = DataLoader(train_set, batch_size=10, num_workers=1)

    n_batches_1 = len(train_loader_1)
    n_batches_2 = len(train_loader_2)
    assert n_batches_1 == n_batches_2

    train_loader_1_ = iterator_(train_loader_1)
    train_loader_2_ = iterator_(train_loader_2)

    output_list1 = []
    output_list2 = []

    for i in range(n_batches_2 + 1):
        data1 = train_loader_1_.__next__()
        data2 = train_loader_2_.__next__()
        output_list1.extend(data1[2])
        output_list2.extend(data2[2])

    assert set(output_list1) == set([Path(x).stem for x in train_loader_1.dataset.filenames['img']])
    assert set(output_list2) == set([Path(x).stem for x in train_loader_2.dataset.filenames['img']])


def test_cityscapes_dataloader():
    augmentations = Compose([Scale(2048), RandomRotate(10)])

    local_path = '../data/Cityscapes'
    dst = CityscapesDataset(local_path, is_transform=True, augmentation=augmentations)
    bs = 4
    trainloader = DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data_samples in enumerate(trainloader):
        [[imgs, labels], _, _] = data_samples

        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        print()


if __name__ == '__main__':
    # test_cityscapes_dataloader()
    test_dataset()
