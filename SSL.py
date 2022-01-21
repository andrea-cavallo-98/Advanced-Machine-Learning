import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
from dataset.cityscapes_dataset import cityscapesDataSet
from model.build_BiSeNet import BiSeNet
import torch

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def ssl(model, save_path, num_classes, batch_size, num_workers, crop_size):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model.eval()
    model.cuda()
    targetloader = data.DataLoader(cityscapesDataSet("Cityscapes", "Cityscapes/train.txt", mean=IMG_MEAN, crop_size=crop_size),
                                   batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                   pin_memory=True)

    predicted_label = np.zeros((len(targetloader), 512, 1024), dtype=np.uint8)
    image_name = []

    classified_pixels = 0.0
    fixed_thres = 0.9

    for index, batch in enumerate(tqdm(targetloader)):
        image, _, name = batch
        output = model(Variable(image).cuda())
        output = nn.functional.softmax(output, dim=1)
        output = nn.functional.upsample(output, (512, 1024), mode='bilinear', align_corners=True).cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)

        label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
        label[prob < fixed_thres] = 255
        classified_pixels += sum(sum(prob >= fixed_thres)) / (1024 * 512)
        predicted_label[index] = np.uint8(label.copy())
        image_name.append(name[0])

    print("Percentage of classified pixels: ", classified_pixels / len(targetloader))

    for index in range(len(targetloader)):
        name = image_name[index]
        output = np.asarray(predicted_label[index], dtype=np.uint8)
        output = Image.fromarray(output)
        name = name.split("/")[1].replace("leftImg8bit", "gtFine_labelIds")
        output.save('%s/%s' % (save_path, name))


if __name__ == '__main__':
  model = BiSeNet(19, 'resnet101')
  model.load_state_dict(torch.load("/content/latest_dice_loss.pth"))
  ssl(model, 'pseudo_labels', 19, 1, 4, (1024, 512))
