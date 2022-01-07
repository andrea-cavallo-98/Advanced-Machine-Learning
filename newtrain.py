import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data, model_zoo
from torch.autograd import Variable
import numpy as np
import pickle
import scipy.misc
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
import torch.cuda.amp as amp
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm
from torchinfo import summary

from model.build_BiSeNet import BiSeNet
from model.discriminator import FCDiscriminator, Lightweight_FCDiscriminator
from loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'BiSeNet'
BATCH_SIZE = 8
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = './GTA5'
DATA_LIST_PATH = './GTA5/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '328,328'
DATA_DIRECTORY_TARGET = './Cityscapes'
DATA_LIST_PATH_TARGET = './Cityscapes/train.txt'
INPUT_SIZE_TARGET = '328,328'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 500 // BATCH_SIZE
NUM_EPOCHS = 50
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
GAN = 'Vanilla'

TARGET = 'cityscapes'
SET = 'train'


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=None,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                        help="Number of epochs.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
    parser.add_argument("--discriminator", type=str, default='standard',
                        help="choose the discriminator.")
    parser.add_argument("--pretrained-model-path", type=str, default=None,
                        help="choose pretrained-model-path")
    parser.add_argument("--pretrained-discriminator-path", type=str, default=None,
                        help="choose pretrained-discriminator-path.")
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main():
    """Create the model and start the training."""

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True
    gpu = args.gpu

    # Create network
    if args.model == 'BiSeNet':
        model = BiSeNet(args.num_classes, 'resnet101')
        """if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                # print i_parts
        model.load_state_dict(new_params)"""
        if args.pretrained_model_path is not None:
            print('load model from %s ...' % args.pretrained_model_path)
            model.module.load_state_dict(torch.load(args.pretrained_model_path))
            print('Done!')

    cudnn.benchmark = True

    # init D
    if args.discriminator == 'standard':
        model_D1 = FCDiscriminator(num_classes=args.num_classes)
    else:
        model_D1 = Lightweight_FCDiscriminator(num_classes=args.num_classes)
    if args.pretrained_discriminator_path is not None:
        print('load model from %s ...' % args.pretrained_discriminator_path)
        model.module.load_state_dict(torch.load(args.pretrained_discriminator_path))
        print('Done!')

    model.eval().cuda()
    model_D1.eval().cuda()
    summary(model, (BATCH_SIZE, 3, 328, 328))
    summary(model_D1, (BATCH_SIZE, 20, 328, 328))

    model.train()
    model.cuda(args.gpu)
    model_D1.train()
    model_D1.cuda(args.gpu)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    for e_epoch in range(args.num_epochs):

        trainloader = data.DataLoader(
            GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                        crop_size=input_size,
                        scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        trainloader_iter = enumerate(trainloader)

        targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                         max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                         crop_size=input_size_target,
                                                         scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                         set=args.set),
                                       batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                       pin_memory=True)

        targetloader_iter = enumerate(targetloader)

        # implement model.optim_parameters(args) to handle different models' lr setting

        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer.zero_grad()

        optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
        optimizer_D1.zero_grad()

        if args.gan == 'Vanilla':
            bce_loss = torch.nn.BCEWithLogitsLoss()
        elif args.gan == 'LS':
            bce_loss = torch.nn.MSELoss()

        interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
        interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear')

        # labels for adversarial training
        source_label = 0
        target_label = 1

        tq = tqdm(total=args.num_steps * args.batch_size)
        tq.set_description('epoch %d' % (e_epoch))

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        for i_iter in range(args.num_steps):

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, i_iter)

            optimizer_D1.zero_grad()
            adjust_learning_rate_D(optimizer_D1, i_iter)

            for sub_i in range(args.iter_size):

                # train G

                # don't accumulate grads in D
                for param in model_D1.parameters():
                    param.requires_grad = False

                # train with source

                _, batch = trainloader_iter.__next__()
                images, labels = batch
                images = Variable(images).cuda(args.gpu)

                # with amp.autocast():
                pred1, pred2, pred3 = model(images)
                # pred1 = interp(pred1)
                # pred2 = interp(pred2)
                # pred3 = interp(pred3)
                loss1 = loss_calc(pred1, labels, args.gpu)
                # loss2 = loss_calc(pred2, labels, args.gpu)
                # loss3 = loss_calc(pred3, labels, args.gpu)
                loss = loss1

                """pred1, _, _ = model(images)
                pred1 = interp(pred1)

                loss = loss_calc(pred1, labels, args.gpu)"""

                # proper normalization
                loss = loss / args.iter_size
                loss.backward()
                loss_seg_value1 += loss.data.cpu()

                # train with target

                _, batch = targetloader_iter.__next__()
                images, _, _ = batch
                images = Variable(images).cuda(args.gpu)

                pred_target1, _, _ = model(images)
                # pred_target1 = interp_target(pred_target1)

                D_out1 = model_D1(F.softmax(pred_target1))

                loss_adv_target1 = bce_loss(D_out1,
                                            Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda(
                                                args.gpu))

                loss = loss_adv_target1
                loss = loss / args.iter_size
                loss.backward()
                loss_adv_target_value1 += loss_adv_target1.data.cpu()

                # train D

                # bring back requires_grad
                for param in model_D1.parameters():
                    param.requires_grad = True

                # train with source
                pred1 = pred1.detach()

                D_out1 = model_D1(F.softmax(pred1))

                loss_D1 = bce_loss(D_out1,
                                   Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda(args.gpu))

                loss_D1 = loss_D1 / args.iter_size / 2

                loss_D1.backward()

                loss_D_value1 += loss_D1.data.cpu()

                # train with target
                pred_target1 = pred_target1.detach()

                D_out1 = model_D1(F.softmax(pred_target1))

                loss_D1 = bce_loss(D_out1,
                                   Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda(args.gpu))

                loss_D1 = loss_D1 / args.iter_size / 2

                loss_D1.backward()

                loss_D_value1 += loss_D1.data.cpu()

            optimizer.step()
            optimizer_D1.step()

            tq.update(args.batch_size)

        tq.close()
        print(
            'epoch = {0} loss_seg1 = {1:.3f}  loss_adv1 = {2:.3f},  loss_D1 = {3:.3f} '.format(
                e_epoch, loss_seg_value1 / args.num_steps, loss_adv_target_value1 / args.num_steps,
                         loss_D_value1 / args.num_steps))

        if (e_epoch % args.save_pred_every == 0 and e_epoch != 0) or e_epoch == args.num_epochs - 1:
            print('taking snapshot ...')
            torch.save(model.module.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D1.module.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D1.pth'))


if __name__ == '__main__':
    main()