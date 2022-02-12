import torch
import argparse
import os
from torch.utils.data import DataLoader
from model.build_BiSeNet import BiSeNet
import numpy as np
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, cal_miou
import tqdm
from dataset.cityscapes_dataset import cityscapesDataSet
from matplotlib import pyplot as plt

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

labels = ["road","sidewalk", "building","wall","fence","pole","light",
    "sign","vegetation","terrain","sky","person","rider","car","truck",
    "bus","train","motocycle","bicycle"]

def save_plot_per_class(hist):
  """
  Create a plot with per-class precision, recall and IoU of predictions
  """
  precision = np.diag(hist) / (hist.sum(0) + 1e-5)
  recall = np.diag(hist) / (hist.sum(1) + 1e-5)

  plt.xticks(range(19), labels, rotation=45)
  plt.plot(precision, '-o')
  plt.plot(recall, '-o')
  plt.plot((np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-5), '-o')
  plt.legend(["Precision", "Recall", "Per-class IoU"])
  plt.gcf().subplots_adjust(bottom=0.2)
  plt.savefig("prec_DA_08")


def eval(model, dataloader, args):
    """
    Perform evaluation of model on test dataset
    """
    print('start test!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        tq = tqdm.tqdm(total=len(dataloader) * args.batch_size)
        tq.set_description('test')
        hist = np.zeros((args.num_classes, args.num_classes))
        for _, (data, label, _) in enumerate(dataloader):
            tq.update(args.batch_size)
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            label = label.squeeze()
            label = np.array(label.cpu())

            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            precision_record.append(precision)
        
        save_plot_per_class(hist)
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)

        tq.close()
        print('precision for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        return precision


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None, required=True,
                        help='The path to the pretrained weights of model')
    parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=1024, help='Width of cropped/resized input image to network')
    parser.add_argument('--data', type=str, default='/path/to/data', help='Path of training data')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')
    args = parser.parse_args(params)

    # Prepare Pytorch train/test Datasets
    test_dataset = cityscapesDataSet("Cityscapes", "Cityscapes/val.txt", augment=False, crop_size=(args.crop_width, args.crop_height), mean=IMG_MEAN)

    # Check dataset sizes
    print('Test Dataset: {}'.format(len(test_dataset)))

    # Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and shuffling)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model if exists
    print('load model from %s ...' % args.checkpoint_path)
    model.module.load_state_dict(torch.load(args.checkpoint_path))
    print('Done!')

    # test
    eval(model, test_dataloader, args)


if __name__ == '__main__':
    params = [
        '--checkpoint_path', '/content/GTA5_124.pth',
        '--data', '/path/to/data',
        '--cuda', '0',
        '--context_path', 'resnet101',
        '--num_classes', '19',
        '--loss', 'crossentropy'
    ]
    main(params)
