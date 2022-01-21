import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from PIL import Image


class GTA5DataSet(data.Dataset):
    def __init__(self, root, list_path, crop_size=(321, 321), augment=False, mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.mean = mean
        self.augment = augment
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []

        for name in self.img_ids:
            img_file = self.root + "/images/" + name
            label_file = self.root + "/labels/" + name
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

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

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        
        label = self.encode_labels(label)

        image -= self.mean
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy()

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
    dst = GTA5DataSet("./GTA5", "./GTA5/train.txt")
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.savefig("image")