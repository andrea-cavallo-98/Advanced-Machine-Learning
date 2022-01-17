import cv2
import argparse
from model.build_BiSeNet import BiSeNet
import os
import torch
import cv2
from imgaug import augmenters as iaa
from PIL import Image
from torchvision import transforms
import numpy as np
from utils import reverse_one_hot, get_label_info, colour_code_segmentation


def encode_labels(self, mask):
    mapping_20 = [[0, 255],[1, 255],[2, 255],[3, 255],[4, 255],[5, 255],
    [6, 255],[7, 0],[8, 1],[9, 255],[10, 255],[11, 2],[12, 3],[13, 4],
    [14, 255],[15, 255],[16, 255],[17, 5],[18, 255],[19, 6],[20, 7],[21, 8],[22, 9],
    [23, 10],[24, 11],[25, 12],[26, 13],[27, 14],[28, 15],[29, 255],
    [30, 255],[31, 16],[32, 17],[33, 18],[-1, 255]]

    label_mask = np.zeros_like(mask)
    for k in mapping_20:
        label_mask[mask == k[0]] = k[1]
    return label_mask


def print_ground_truth(image_path, label_path):
    # read csv label path
    label_info = get_label_info()

    label = Image.open(label_path)
    label = label.resize((328, 328), Image.NEAREST)
    
    label = np.asarray(label, np.float32)
    label = encode_labels(None, label)

    label = colour_code_segmentation(np.array(label), label_info)
    label = cv2.resize(np.uint8(label), (960, 720), interpolation=cv2.INTER_NEAREST)
    image = cv2.imread(image_path)
    image = cv2.resize(np.uint8(image), (960, 720), interpolation=cv2.INTER_LINEAR)
    label = cv2.cvtColor(np.uint8(label), cv2.COLOR_RGB2BGR)
    added_image = cv2.addWeighted(image,0.4,label,0.5,0)
    cv2.imwrite('demo_images/' + image_path.split("/")[-1], added_image)


def predict_on_image(model, data):
    # pre-processing on image
    image = Image.open(data).convert('RGB')
    image = image.resize((328, 328), Image.BILINEAR)
    image = np.asarray(image, np.float32)
    size = image.shape
    image = image[:, :, ::-1]  # change to BGR
    image = image - np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    image = image.transpose((2, 0, 1))
    image = torch.tensor(image).unsqueeze(0).cuda()
    # read csv label path
    label_info = get_label_info()
    # predict
    model.eval()
    predict = model(image).squeeze()
    predict = reverse_one_hot(predict.cpu())
    print("Prediction : ", predict)
    predict = colour_code_segmentation(np.array(predict), label_info)
    predict = cv2.resize(np.uint8(predict), (960, 720))
    image = cv2.imread(data)
    image = cv2.resize(np.uint8(image), (960, 720))
    predict = cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR)
    added_image = cv2.addWeighted(image,0.4,predict,0.5,0)
    cv2.imwrite('demo_images/aachen.png', added_image)

def main():
  # build model
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  model = BiSeNet(19, 'resnet101')
  if torch.cuda.is_available():
      model = torch.nn.DataParallel(model).cuda()

  # load pretrained model if exists
  print('load model from %s ...' % './checkpoints_18_sgd/latest_dice_loss.pth')
  model.module.load_state_dict(torch.load('./checkpoints_18_sgd/latest_dice_loss.pth'))
  print('Done!')

  #predict_on_image(model, 'Cityscapes/images/aachen_000001_000019_leftImg8bit.png')
  predict_on_image(model, 'Cityscapes/images/bremen_000253_000019_leftImg8bit.png')

if __name__ == '__main__':
  main()