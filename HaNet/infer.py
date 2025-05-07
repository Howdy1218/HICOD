import os
import time
import datetime
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from skimage.transform import resize
import h5py
from numpy import mean
from config import *
from misc import *
from HaNet import HaNet
from joint_transform import transform_hsi  # Assuming you have this module

torch.manual_seed(2024)
device_ids = [0]
torch.cuda.set_device(device_ids[0])

results_path = './results'
check_mkdir(results_path)
exp_name = 'HaNet'
args = {
    'scale': 224,
    'save_results': True
}

print(torch.__version__)

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

hsi_transform = transform_hsi(args['scale'])

to_pil = transforms.ToPILImage()

to_test = OrderedDict([
    ('CHAMELEON', chameleon_path),
    ('CAMO', camo_path),
    ('COD10K', cod10k_path),
    ('NC4K', nc4k_path)
])

results = OrderedDict()


def read_mat_file(file_path):
    try:
        with h5py.File(file_path, 'r') as file:
            hsi = np.array(file['cube'])  # 假设数据存储在键 'hsi' 下
            return hsi
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def main():
    net = HaNet(backbone_path).cuda(device_ids[0])

    net.load_state_dict(torch.load('D:\\HaNet\\ckpt\\HaNet\\45.pth'))
    print('Load {} succeed!'.format('50.pth'))

    net.eval()
    with torch.no_grad():
        start = time.time()
        for name, root in to_test.items():
            time_list = []
            image_path = os.path.join(root, 'imgs')
            #hsi_path = os.path.join(root, 'hsis')
            hsi_path = ""

            if args['save_results']:
                check_mkdir(os.path.join(results_path, exp_name, name))

            img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('jpg')]
            for idx, img_name in enumerate(img_list):
                img = Image.open(os.path.join(image_path, img_name + '.jpg')).convert('RGB')

                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda(device_ids[0])

                # 构建HSI文件路径
                hsi_file_path = os.path.join(hsi_path, img_name + '.mat')
                if not os.path.isfile(hsi_file_path):
                    print(f"HSI file does not exist: {hsi_file_path}")
                    continue

                # 使用 h5py 加载 HSI 数据
                hsi_data = read_mat_file(hsi_file_path)
                if hsi_data is None:
                    continue

                hsi_var = Variable(hsi_transform(hsi_data).unsqueeze(0)).cuda(device_ids[0])

                start_each = time.time()
                _, _, _, prediction = net(img_var, hsi_var)
                time_each = time.time() - start_each
                time_list.append(time_each)

                prediction = np.array(transforms.Resize((h, w))(to_pil(prediction.data.squeeze(0).cpu())))

                if args['save_results']:
                    Image.fromarray(prediction).convert('L').save(
                        os.path.join(results_path, exp_name, name, img_name + '.png'))

            print(('{}'.format(exp_name)))
            print("{}'s average Time Is : {:.3f} s".format(name, mean(time_list)))
            print("{}'s average Time Is : {:.1f} fps".format(name, 1 / mean(time_list)))

    end = time.time()
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))


if __name__ == '__main__':
    main()
