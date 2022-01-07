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