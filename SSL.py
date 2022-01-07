from numpy.lib.npyio import _savez_compressed_dispatcher
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from PIL import Image
from tqdm import tqdm
import json
import os.path as osp
import os
import numpy as np
from model.build_BiSeNet import BiSeNet
from dataset.cityscapes_dataset import cityscapesDataSet

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


def ssl(model, save_path, num_classes, batch_size, num_workers):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = BiSeNet(num_classes, 'resnet101')

    model.eval()
    model.cuda()
    targetloader = data.DataLoader(cityscapesDataSet("Cityscapes", "Cityscapes/train.txt", mean=IMG_MEAN),
                                   batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                   pin_memory=True)

    predicted_label = np.zeros((len(targetloader), 512, 1024))
    predicted_prob = np.zeros((len(targetloader), 512, 1024))
    image_name = []

    for index, batch in enumerate(tqdm(targetloader)):
        image, _, name = batch
        output = model(Variable(image).cuda())
        output = nn.functional.softmax(output, dim=1)
        output = nn.functional.upsample(output, (512, 1024), mode='bilinear', align_corners=True).cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)

        label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
        predicted_label[index] = label.copy()
        predicted_prob[index] = prob.copy()
        image_name.append(name[0])

    thres = []
    for i in range(num_classes):
        x = predicted_prob[predicted_label == i]
        if len(x) == 0:
            thres.append(0)
            continue
        x = np.sort(x)
        thres.append(x[np.int(np.round(len(x) * 0.5))])
    print(thres)
    thres = np.array(thres)
    thres[thres > 0.9] = 0.9
    print(thres)
    for index in range(len(targetloader)):
        name = image_name[index]
        label = predicted_label[index]
        prob = predicted_prob[index]
        for i in range(num_classes):
            label[(prob < thres[i]) * (label == i)] = 255
        output = np.asarray(label, dtype=np.uint8)
        output = Image.fromarray(output)
        name = name.split("/")[1].replace("leftImg8bit", "gtFine_labelIds")
        output.save('%s/%s' % (save_path, name))


if __name__ == '__main__':
    ssl(None, 'pseudo_labels', 20, 1, 4)