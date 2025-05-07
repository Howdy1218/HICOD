import os

backbone_path = './backbone/resnet/resnet50-19c8e357.pth'

datasets_root = 'D:\\HaNet\\dataset'


cod_training_root = os.path.join(datasets_root, 'train')

chameleon_path = os.path.join(datasets_root, 'test\\CHAMELEON')
camo_path = os.path.join(datasets_root, 'test\\CAMO')
cod10k_path = os.path.join(datasets_root, 'test\\COD10K')
nc4k_path = os.path.join(datasets_root, 'test\\NC4K')