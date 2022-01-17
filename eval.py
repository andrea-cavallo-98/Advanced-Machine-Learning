import torch
import argparse
import os
from torch.utils.data import DataLoader
from model.build_BiSeNet import BiSeNet
import numpy as np
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, cal_miou
import tqdm
from dataset.cityscapes_dataset import cityscapesDataSet

IMG_MEAN = np.array((73.158359210711552,82.908917542625858,72.392398761941593), dtype=np.float32)

def eval(model, dataloader, args, csv_path):
    print('start test!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        tq = tqdm.tqdm(total=len(dataloader) * args.batch_size)
        tq.set_description('test')
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label, _) in enumerate(dataloader):
            tq.update(args.batch_size)
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())
            # predict = colour_code_segmentation(np.array(predict), label_info)

            label = label.squeeze()
            # if args.loss == 'dice':
            # label = reverse_one_hot(label)
            label = np.array(label.cpu())
            # label = colour_code_segmentation(np.array(label), label_info)

            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            precision_record.append(precision)
        precision = np.mean(precision_record)
        #miou_list = per_class_iu(hist)[:-1]
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)

        # miou_dict, miou = cal_miou(miou_list, csv_path)
        # print('IoU for each class:')
        # for key in miou_dict:
        #    print('{}:{},'.format(key, miou_dict[key]))
        tq.close()
        print('precision for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        return precision


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None, required=True,
                        help='The path to the pretrained weights of model')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')
    parser.add_argument('--data', type=str, default='/path/to/data', help='Path of training data')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')
    args = parser.parse_args(params)

    # Prepare Pytorch train/test Datasets
    test_dataset = cityscapesDataSet("Cityscapes", "Cityscapes/val.txt", augment=False, mean=IMG_MEAN)

    # Check dataset sizes
    print('Test Dataset: {}'.format(len(test_dataset)))

    # Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and shuffling)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model if exists
    print('load model from %s ...' % args.checkpoint_path)
    model.module.load_state_dict(torch.load(args.checkpoint_path))
    print('Done!')

    # get label info
    # label_info = get_label_info(csv_path)
    # test
    eval(model, test_dataloader, args, "file.csv")


if __name__ == '__main__':
    params = [
        '--checkpoint_path', './checkpoints_18_sgd/latest_dice_loss.pth',
        '--data', '/path/to/data',
        '--cuda', '0',
        '--context_path', 'resnet101',
        '--num_classes', '19',
        '--loss', 'crossentropy'
    ]
    main(params)
