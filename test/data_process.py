import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import ToPILImage

class LabelProcessor:   # 对应data process and load.ipynb 1.处理标签文件中colormap的数据
    """对标签图像的编码"""

    def __init__(self, file_path):

        self.colormap = self.read_color_map(file_path)

        self.cm2lbl = self.encode_label_pix(self.colormap)

    # 静态方法装饰器， 可以理解为定义在类中的普通函数，可以用self.<name>方式调用
    # 在静态方法内部不可以示例属性和实列对象，即不可以调用self.相关的内容
    # 使用静态方法的原因之一是程序设计的需要（简洁代码，封装功能等）
    @staticmethod
    def read_color_map(file_path):  # data process and load.ipynb: 处理标签文件中colormap的数据
        pd_label_color = pd.read_csv(file_path, sep=',')
        colormap = []
        for i in range(len(pd_label_color.index)):
            tmp = pd_label_color.iloc[i]
            color = [tmp['r'], tmp['g'], tmp['b']]
            colormap.append(color)
        return colormap

    @staticmethod
    def encode_label_pix(colormap):     # data process and load.ipynb: 标签编码，返回哈希表
        cm2lbl = np.zeros(256 ** 3)  #哈希表，一对一 或多对一的查找方式，加快查询方式，
        for i, cm in enumerate(colormap): #
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return cm2lbl#哈希映射表

#目的是为了把标签形式转化为对应的类别，通过哈希映射，每一个类别对应原始像素的每一个像素点，
    def encode_label_img(self, img):

        data = np.array(img, dtype='int32')#转成numpy格式，
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]#这是一片数字
        return np.array(self.cm2lbl[idx], dtype='int64')    #
label_processor = LabelProcessor("class_dict.csv")#csv文件



path = r"E:\PycharmProject\FCN\Datasets\ACDC-all\val\gt"
files = os.listdir(path)
for file in files:
    name = os.path.join(path,file)
    label = Image.open(name)
    label = np.array(label)
    label = label_processor.encode_label_img(label)# 对标签编码
    label = np.array(label,dtype=np.uint8)
    label =Image.fromarray(label)
    label.save("E:/PycharmProject/FCN/Datasets/ACDC-all/val/gtt/{}".format(file))





