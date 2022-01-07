import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image


class cityscapesDataSet(data.Dataset):

    def __init__(self, root, list_path, augment=False, max_iters=None, crop_size=(328, 328), mean=(128, 128, 128),
                 scale=True, mirror=True, ignore_label=255, set='val', pseudo_labels_path=None):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.augment = augment
        self.pseudo_labels_path = pseudo_labels_path
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        # if not max_iters==None:
        #    self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            # img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            img_file = self.root + "/images/" + name.split("/")[1]
            if pseudo_labels_path is None:
                label = self.root + "/labels/" + name.split("/")[1].replace("leftImg8bit", "gtFine_labelIds")
            else:
                label = pseudo_labels_path + name.split("/")[1].replace("leftImg8bit", "gtFine_labelIds")
            self.files.append({
                "img": img_file,
                "label": label,
                "name": name
            })

    def __len__(self):
        return len(self.files)

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

    def __getitem__(self, index):

        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])

        if self.augment:
            AUG_PROB = 0.5

            if np.random.rand() < AUG_PROB:
                hflip_t = torchvision.transforms.RandomHorizontalFlip(p=1)

                image = hflip_t(image)
                label = hflip_t(label)

        name = datafiles["name"]

        # print("Initial shape of label: ", np.array(label).shape)
        # resize
        image = image.resize(self.crop_size, Image.BILINEAR)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label)

        label = self.encode_labels(label)

        complete_labels = label

        """if self.pseudo_labels_path is None:
            complete_labels = []
            for c in range(19):
                current_mask = 1 * (label == c)
                complete_labels.append(current_mask)

            complete_labels = np.array(complete_labels)
        else:
            complete_labels = label"""

        # print(complete_labels.shape)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), complete_labels.copy(), name


if __name__ == '__main__':
    """
    dst = GTA5DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
    """
    image = Image.open("/content/Cityscapes/images/aachen_000001_000019_leftImg8bit.png").convert('RGB')
    label = Image.open("/content/Cityscapes/labels/aachen_000001_000019_gtFine_labelIds.png")
    img = np.asarray(image)
    plt.imshow(img)
    plt.savefig("original_image")
    img = np.asarray(label)
    plt.imshow(img)
    plt.savefig("original_label")

    hflip_t = torchvision.transforms.RandomHorizontalFlip(p=1)
    image = hflip_t(image)
    label = hflip_t(label)
    img = np.asarray(image)
    plt.imshow(img)
    plt.savefig("augmented_image")
    img = np.asarray(label)
    plt.imshow(img)
    plt.savefig("augmented_label")






