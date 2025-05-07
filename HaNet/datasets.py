import os
import os.path
import h5py
import torch.utils.data as data
import torchvision.transforms
from PIL import Image
import torchvision.transforms as tf
from utils import joint_transform
import joint_transform
from joint_transform import transform_hsi
def read_mat_file(file_path):
    # 打开HDF5文件
    with h5py.File(file_path, 'r') as file:
        data = file['cube'][:]
        return data
joint_transform = joint_transform.Compose([
    joint_transform.RandomHorizontallyFlip(),
    joint_transform.Resize((224, 224))
])
def make_dataset(root):
    image_path = os.path.join(root, 'image')
    mask_path = os.path.join(root, 'mask')
    hsi_path = os.path.join(root, 'hsi')
    img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.jpg')]
    return [(os.path.join(image_path, img_name + '.jpg'),
             os.path.join(mask_path, img_name + '.png'),
             os.path.join(hsi_path, img_name + '.mat')) for img_name in img_list]

class ImageFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, joint_transform_rm=None, transform_rgb=None, transform_hsi =None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform_rm = joint_transform_rm
        self.transform_rgb = transform_rgb
        self.transform_hsi = transform_hsi
        self.target_transform = target_transform
        self.rmf = read_mat_file

    def __getitem__(self, index):
        img_path, gt_path, hsi_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB') # (C,W,H)
        target = Image.open(gt_path).convert('L')
        hsi = self.rmf(hsi_path)  # (C,W,H)
        if self.joint_transform_rm is not None:
            img, target= self.joint_transform_rm(img, target)
        if self.transform_rgb is not None:
            img = self.transform_rgb(img)
        if self.transform_hsi is not None:
            hsi = self.transform_hsi(hsi)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, hsi # img:(c, w, h) target:(c, w, h) hsi:(c,w,h)

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    root = 'D:\\HaNet\\dataset\\train'
    img, target, hsi = ImageFolder(root).__getitem__(0)
    totensor = torchvision.transforms.ToTensor()
    img = totensor(img)
    # print('img经过Totensor的形状：',img.shape) （c,h,w)
    # print(hsi.shape)# (c,w,h)
    # print(img.size)
    Tf_h = transform_hsi(224)
    hsi = Tf_h(hsi)
    # print(hsi.shape)