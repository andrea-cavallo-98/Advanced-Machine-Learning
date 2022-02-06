import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data, model_zoo
from torch.autograd import Variable
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
from torchinfo import summary
from utils import poly_lr_scheduler


from model.build_BiSeNet import BiSeNet
from model.discriminator import FCDiscriminator, Lightweight_FCDiscriminator
from loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'BiSeNet'
BATCH_SIZE = 4
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = './GTA5'
DATA_LIST_PATH = './GTA5/train.txt'
INPUT_SIZE = '1024, 512'
DATA_DIRECTORY_TARGET = './Cityscapes'
DATA_LIST_PATH_TARGET = './Cityscapes/train.txt'
INPUT_SIZE_TARGET = '1024, 512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 500 // BATCH_SIZE
NUM_EPOCHS = 50
POWER = 0.9
RANDOM_SEED = 1234
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
DISCRIMINATOR = "standard" # standard or lightweight
PRETRAINED_MODEL_PATH = None
PRETRAINED_DISCRIMINATOR_PATH = None
INITIAL_EPOCH = 0
LEARNING_RATE_D = 1e-4
GAN = 'Vanilla'


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
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
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
    parser.add_argument("--discriminator", type=str, default=DISCRIMINATOR,
                        help="choose the discriminator.")
    parser.add_argument("--pretrained-model-path", type=str, default=PRETRAINED_MODEL_PATH,
                        help="choose pretrained-model-path")
    parser.add_argument("--pretrained-discriminator-path", type=str, default=PRETRAINED_DISCRIMINATOR_PATH,
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


def main():
    """Create the model and start the training."""

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True

    # Create network
    if args.model == 'BiSeNet':
        model = BiSeNet(args.num_classes, 'resnet101')
        if args.pretrained_model_path is not None:
            print('load model from %s ...' % args.pretrained_model_path)
            model.load_state_dict(torch.load(args.pretrained_model_path))
            print('Done!')

    cudnn.benchmark = True

    # init D
    if args.discriminator == 'standard':
        model_D = FCDiscriminator(num_classes=args.num_classes)
    else:
        model_D = Lightweight_FCDiscriminator(num_classes=args.num_classes)
    if args.pretrained_discriminator_path is not None:
        print('load model from %s ...' % args.pretrained_discriminator_path)
        model.module.load_state_dict(torch.load(args.pretrained_discriminator_path))
        print('Done!')

    ## Produce summary of models
    model.eval().cuda()
    model_D.eval().cuda()
    #summary(model, (BATCH_SIZE, 3, 1024, 512))
    #summary(model_D, (BATCH_SIZE, 19, 1024, 512))

    model.train()
    model.cuda(args.gpu)
    model_D.train()
    model_D.cuda(args.gpu)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    for epoch in range(INITIAL_EPOCH, args.num_epochs):

        trainloader = data.DataLoader(
            GTA5DataSet(args.data_dir, args.data_list, crop_size=input_size, augment = True, mean=IMG_MEAN),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        trainloader_iter = enumerate(trainloader)

        targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target, 
                                                        crop_size=input_size_target, mean=IMG_MEAN),
                                       batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                       pin_memory=True)

        targetloader_iter = enumerate(targetloader)

        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer.zero_grad()

        optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
        optimizer_D.zero_grad()

        if args.gan == 'Vanilla':
            bce_loss = torch.nn.BCEWithLogitsLoss()
        elif args.gan == 'LS':
            bce_loss = torch.nn.MSELoss()

        # labels for adversarial training
        source_label = 0
        target_label = 1

        tq = tqdm(total=args.num_steps * args.batch_size)
        tq.set_description('epoch %d' % (epoch))

        loss_seg_value = 0
        loss_adv_target_value = 0
        loss_D_value = 0

        for i_iter in range(args.num_steps):

            optimizer.zero_grad()
            poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)

            optimizer_D.zero_grad()
            poly_lr_scheduler(optimizer_D, args.learning_rate_D, iter=epoch, max_iter=args.num_epochs)

            for _ in range(args.iter_size):

                # train G

                # don't accumulate grads in D
                for param in model_D.parameters():
                    param.requires_grad = False

                # train with source

                _, batch = trainloader_iter.__next__()
                images, labels = batch
                images = Variable(images).cuda(args.gpu)

                pred, pred1, pred2 = model(images)
                loss1 = loss_calc(pred, labels, args.gpu)
                loss2 = loss_calc(pred1, labels, args.gpu)
                loss3 = loss_calc(pred2, labels, args.gpu)
                
                # proper normalization
                loss = (loss1 + loss2 + loss3) / args.iter_size
                loss.backward()
                loss_seg_value += loss.data.cpu()

                # train with target

                _, batch = targetloader_iter.__next__()
                images, _, _ = batch
                images = Variable(images).cuda(args.gpu)

                pred_target, _, _ = model(images)

                D_out = model_D(F.softmax(pred_target))
                loss_adv_target = bce_loss(D_out, 
                                  Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda(args.gpu))

                loss = loss_adv_target
                loss = loss / args.iter_size
                loss.backward()
                loss_adv_target_value += loss_adv_target.data.cpu()

                # train D

                # bring back requires_grad
                for param in model_D.parameters():
                    param.requires_grad = True

                # train with source
                pred = pred.detach()

                D_out = model_D(F.softmax(pred))

                loss_D = bce_loss(D_out,
                                   Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda(args.gpu))

                loss_D = loss_D / args.iter_size / 2

                loss_D.backward()

                loss_D_value += loss_D.data.cpu()

                # train with target
                pred_target = pred_target.detach()

                D_out = model_D(F.softmax(pred_target))

                loss_D = bce_loss(D_out,
                                   Variable(torch.FloatTensor(D_out.data.size()).fill_(target_label)).cuda(args.gpu))

                loss_D = loss_D / args.iter_size / 2

                loss_D.backward()

                loss_D_value += loss_D.data.cpu()

            optimizer.step()
            optimizer_D.step()

            tq.update(args.batch_size)

        tq.close()
        print(
            'epoch = {0} loss_seg = {1:.3f}  loss_adv = {2:.3f},  loss_D = {3:.3f} '.format(
                epoch, loss_seg_value / args.num_steps, loss_adv_target_value / args.num_steps,
                         loss_D_value / args.num_steps))

        if ((epoch + 1) % args.save_pred_every == 0 and epoch != 0) or epoch == args.num_epochs - 1:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D.pth'))


if __name__ == '__main__':
    main()