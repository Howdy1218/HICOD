import os
import os.path
import h5py
import random
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as tf

from skimage.transform import resize
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)  PIL: (w, h)

    def __call__(self, img, mask,):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)

class transform_hsi(data.Dataset):
    def __init__(self, scale):
        super(transform_hsi, self).__init__()
        self.ToTensor = tf.ToTensor()
        self.scale = scale
    def __call__(self, hsi):
        print("Original shape:", hsi.shape)# (c,w,h)
        hsi = np.transpose(hsi, (0, 2, 1))# (c,h,w)
        print(hsi.shape)
        hsi = resize(hsi, (31, self.scale, self.scale), order=1, mode='reflect', anti_aliasing=True)#(c,h,w)
        print("After resize:", hsi.shape)

        # 转换为 HWC 格式以符合 ToTensor 的预期
        hsi = np.transpose(hsi, (1, 2, 0))  # CHW to HWC (h,w,c)
        #print("After transpose to HWC:", hsi.shape)

        hsi = self.ToTensor(hsi)  # 现在是 CxHxW，因为输入是 HWC
        #print("After ToTensor:", hsi.shape)

        return hsi