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

import cv2
from utils import get_label_info, colour_code_segmentation

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


def ssl(model, save_path, num_classes, batch_size, num_workers, gpu, crop_size):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model.eval()
    model.cuda(gpu)
    targetloader = data.DataLoader(
        cityscapesDataSet("Cityscapes", "Cityscapes/train.txt", mean=IMG_MEAN, crop_size=crop_size),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True)

    predicted_label = np.zeros((len(targetloader), 512, 1024), dtype=np.uint8)
    image_name = []
    images = []

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
        images.append(image)

    print("Percentage of classified pixels: ", classified_pixels / len(targetloader))

    for index in range(len(targetloader)):
        name = image_name[index]
        output = np.asarray(predicted_label[index], dtype=np.uint8)
        output = Image.fromarray(output)
        name = name.split("/")[1].replace("leftImg8bit", "gtFine_labelIds")
        output.save('%s/%s' % (save_path, name))

        # display output and image
        image = np.asarray(images[index], np.float32)
        image = np.transpose(image.squeeze(), (1, 2, 0))
        image = IMG_MEAN + image.squeeze()
        image = image[:, :, ::-1]
        image = cv2.resize(np.uint8(image), (960, 720))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        label_info = get_label_info()
        output = np.asarray(output, np.float32)
        output = colour_code_segmentation(np.array(output.squeeze()), label_info)
        output = cv2.resize(np.uint8(output), (960, 720))
        output = cv2.cvtColor(np.uint8(output), cv2.COLOR_RGB2BGR)

        added_image = cv2.addWeighted(image, 0.4, output, 0.5, 0)
        cv2.imwrite('demo_images/' + name.split("/")[-1], added_image)


if __name__ == '__main__':
    model = BiSeNet(19, 'resnet101')
    model.load_state_dict(torch.load("/content/latest_dice_loss.pth"))
    ssl(model, 'pseudo_labels', 19, 1, 4, (1024, 512))
