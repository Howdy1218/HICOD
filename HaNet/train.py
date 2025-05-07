import datetime
import time
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

import joint_transform
from config import cod_training_root
from config import backbone_path
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from HaNet import HaNet
from joint_transform import transform_hsi
import loss

cudnn.benchmark = True

torch.manual_seed(2024)
device_ids = [0]

ckpt_path = './ckpt'
exp_name = 'HaNet_CM_poly'

args = {
    'epoch_num': 40,
    'train_batch_size': 16,
    'last_epoch': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 448,
    'save_point': [],
    'poly_train': True,
    'optimizer': 'SGD',
}

# Path.
check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
vis_path = os.path.join(ckpt_path, exp_name, 'log')
check_mkdir(vis_path)
log_path = os.path.join(ckpt_path, exp_name, datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + '.txt')
writer = SummaryWriter(log_dir=vis_path, comment=exp_name)

# Transform Data.
joint_transform_rm = joint_transform.Compose([
    joint_transform.RandomHorizontallyFlip(),
    joint_transform.Resize((args['scale'], args['scale']))
])

img_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]) # c h w  (3,224,224)
target_transform = transforms.ToTensor() #c,h,w (1,224,224)
hsi_transform = transform_hsi(args['scale']) # c h w (31, 224,224)

# Prepare Data Set.
train_set = ImageFolder(cod_training_root, joint_transform_rm=joint_transform_rm, transform_rgb=img_transform, transform_hsi = hsi_transform, target_transform=target_transform)
# print("Train set: {}".format(train_set.__len__()))
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=True)
# print(train_loader)
# print(train_set)
total_epoch = args['epoch_num'] * len(train_loader)

# loss function
structure_loss = loss.structure_loss().cuda(device_ids[0])
bce_loss = nn.BCEWithLogitsLoss().cuda(device_ids[0])
iou_loss = loss.IOU().cuda(device_ids[0])

def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + iou_out

    return loss

def main():
    print(args)
    print(exp_name)
    net = HaNet(backbone_path).cuda(device_ids[0]).train()


    if args['optimizer'] == 'Adam':
        print("Adam")
        optimizer = optim.Adam([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ])
    else:
        print("SGD")
        optimizer = optim.SGD([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('Training Resumes From \'%s\'' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        total_epoch = (args['epoch_num'] - int(args['snapshot'])) * len(train_loader)
        print(total_epoch)

    net = nn.DataParallel(net, device_ids=device_ids)
    print("Using {} GPU(s) to Train.".format(len(device_ids)))

    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)
    writer.close()

def train(net, optimizer):
    curr_iter = 1
    start_time = time.time()

    for epoch in range(args['last_epoch'] + 1, args['last_epoch'] + 1 + args['epoch_num']):
        loss_record, loss_1_record, loss_2_record,loss_3_record,loss_4_record,loss_5_record,loss_6_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        train_iterator = tqdm(train_loader, total=len(train_loader))

        for data in train_iterator:
            if args['poly_train']:
                base_lr = args['lr'] * (1 - float(curr_iter) / float(total_epoch)) ** args['lr_decay']
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr

            inputs, labels, hsis = data
            batch_size = inputs.size(0)
            inputs = inputs.cuda(device_ids[0])
            labels = labels.cuda(device_ids[0])
            hsis = hsis.cuda(device_ids[0])
            optimizer.zero_grad()

            predict4_rgb,predict4_hsi, predict3_rgb, predict3_hsi, predict2_rgb, predict2_hsi, predict1 = net(inputs, hsis)

            loss_1 = bce_iou_loss(predict4_rgb, labels)
            loss_2 = structure_loss(predict3_rgb, labels)
            loss_3 = structure_loss(predict3_hsi, labels)
            loss_4 = structure_loss(predict2_rgb, labels)
            loss_5 = structure_loss(predict2_hsi, labels)
            loss_6 = structure_loss(predict1, labels)




            loss = 1 * loss_1 + 1 * loss_2 + 1 * loss_3 + 2* loss_4 + 2 *loss_5 + 4* loss_6

            # loss = 1 * loss_1 + 1 * loss_2 + 2 * loss_3 + 4 * loss_4
            # loss_1 = bce_iou_loss(predict_1, labels)
            # loss_2 = structure_loss(predict_2, labels)
            # loss_3 = structure_loss(predict_3, labels)
            # loss_4 = structure_loss(predict_4, labels)
            #
            # loss = 1 * loss_1 + 1 * loss_2 + 2 * loss_3 + 4 * loss_4

            loss.backward()

            optimizer.step()

            loss_record.update(loss.data, batch_size)
            loss_1_record.update(loss_1.data, batch_size)
            loss_2_record.update(loss_2.data, batch_size)
            loss_3_record.update(loss_3.data, batch_size)
            loss_4_record.update(loss_4.data, batch_size)
            loss_5_record.update(loss_5.data, batch_size)
            loss_6_record.update(loss_6.data, batch_size)


            if curr_iter % 10 == 0:
                writer.add_scalar('loss', loss, curr_iter)
                writer.add_scalar('loss_1', loss_1, curr_iter)
                writer.add_scalar('loss_2', loss_2, curr_iter)
                writer.add_scalar('loss_3', loss_3, curr_iter)
                writer.add_scalar('loss_4', loss_4, curr_iter)
                writer.add_scalar('loss_5', loss_5, curr_iter)
                writer.add_scalar('loss_6', loss_6, curr_iter)


            log = '[%3d], [%6d], [%.6f], [%.5f], [%.5f], [%.5f],  [%.5f], [%.5f], [%.5f],  [%.5f]' % \
                  (epoch, curr_iter, base_lr, loss_record.avg, loss_1_record.avg, loss_2_record.avg,loss_3_record.avg,loss_4_record.avg, loss_5_record.avg,loss_6_record.avg,
                    )
            train_iterator.set_description(log)
            open(log_path, 'a').write(log + '\n')

            curr_iter += 1

        if epoch in args['save_point']:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            net.cuda(device_ids[0])

        if epoch >= args['epoch_num']:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
            print(exp_name)
            print("Optimization Have Done!")
            return

if __name__ == '__main__':
    print(torch.__version__)
    print("Train set: {}".format(train_set.__len__()))
    main()