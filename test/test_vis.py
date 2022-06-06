from generalframework.utils import Writter_tf
from generalframework.dataset import MedicalImageDataset, segment_transform, augment
from generalframework.arch import get_arch
from torch.utils.data import DataLoader
import shutil
from generalframework.loss import dice
import matplotlib.pyplot as plt

plt.switch_backend('agg')


def test_visualization():
    tensorbord_dir = '../tmp/ACDC_cotraining'
    root_dir = 'E:/PycharmProject/FCN/Datasets/ACDC-all'
    train_dataset = MedicalImageDataset(root_dir, 'train', subfolders=['img','gtt'],transform=segment_transform((200, 200)), augment=augment)
    val_dataset = MedicalImageDataset(root_dir, 'val',  subfolders=['img','gtt'],transform=segment_transform((200, 200)), augment=None)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    torchnet = get_arch('enet', {'num_classes': 2})
    writer = Writter_tf(tensorbord_dir, torchnet, 4)

    for i in range(2):
        writer.add_images(train_loader, i)

    ## clean up
    shutil.rmtree(tensorbord_dir)

if __name__ == '__main__':
    test_visualization()