import os
import sys
import cv2
import shutil
import torchvision
import numpy as np
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader

from utils import get_label_info, colour_code_segmentation

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/content/Advanced-Machine-Learning/')

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


def print_image_and_label(dataloader):
    """
    Used for testing the correct working of the dataset class
    """
    label_info = get_label_info()
    for i, (image, label, name) in enumerate(dataloader):
        image = np.asarray(image, np.float32)
        # process image
        image = np.transpose(image.squeeze(), (1, 2, 0))
        image = (128, 128, 128) + image.squeeze()
        image = image[:, :, ::-1]
        image = cv2.resize(np.uint8(image), (960, 720))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # process label
        label = colour_code_segmentation(np.array(label.squeeze()), label_info)
        label = cv2.resize(np.uint8(label), (960, 720))
        label = cv2.cvtColor(np.uint8(label), cv2.COLOR_RGB2BGR)

        added_image = cv2.addWeighted(image, 0.4, label, 0.5, 0)
        cv2.imwrite('demo_images/' + name[0].split("/")[-1], added_image)


class cityscapesDataSet(data.Dataset):

    def __init__(self, root, list_path, augment=False, crop_size=(328, 328), mean=IMG_MEAN, pseudo_labels_path=None):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.mean = mean
        self.augment = augment
        self.pseudo_labels_path = pseudo_labels_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []

        for name in self.img_ids:
            img_file = self.root + "/images/" + name.split("/")[1]
            if self.pseudo_labels_path is None:
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

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        if self.augment:
            AUG_PROB = 0.5
            if np.random.rand() < AUG_PROB:
                hflip_t = torchvision.transforms.RandomHorizontalFlip(p=1)
                image = hflip_t(image)
                label = hflip_t(label)

        image = image.resize(self.crop_size, Image.BILINEAR)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        if self.pseudo_labels_path is None: # don't encode labels if they are already encoded
          label = self.encode_labels(label)

        image -= self.mean
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), name

    def encode_labels(self, mask):
        """
        Maps the initial 34 classes for labels into the 19 used in this work
        @param mask: label on which the mapping is applied
        @return: label with 19 different possible classes
        """
        mapping_19 = [[0, 255], [1, 255], [2, 255], [3, 255], [4, 255], [5, 255], [6, 255], [7, 0], [8, 1], [9, 255],
                      [10, 255], [11, 2], [12, 3], [13, 4], [14, 255], [15, 255], [16, 255], [17, 5], [18, 255],
                      [19, 6], [20, 7], [21, 8], [22, 9], [23, 10], [24, 11], [25, 12], [26, 13], [27, 14], [28, 15],
                      [29, 255], [30, 255], [31, 16], [32, 17], [33, 18], [-1, 255]]

        label_mask = np.zeros_like(mask)
        for k in mapping_19:
            label_mask[mask == k[0]] = k[1]
        return label_mask


if __name__ == '__main__':

    if os.path.exists("demo_images/"):
        shutil.rmtree("demo_images/")
    os.makedirs("demo_images/")

    test_dataset = cityscapesDataSet("Cityscapes", "Cityscapes/val.txt", augment=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    print_image_and_label(test_dataloader)
