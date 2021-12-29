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



    def __init__(self, root, list_path, augment=False, max_iters=None, crop_size=(328, 328), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.augment = augment
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        #if not max_iters==None:
        #    self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            #img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            img_file = self.root + "/images/" + name.split("/")[1]
            label = self.root + "/labels/" + name.split("/")[1].replace("leftImg8bit", "gtFine_labelIds")
            self.files.append({
                "img": img_file,
                "label": label,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def encode_labels(self, mask):
        mapping_20 = { 
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 1,
                8: 2,
                9: 0,
                10: 0,
                11: 3,
                12: 4,
                13: 5,
                14: 0,
                15: 0,
                16: 0,
                17: 6,
                18: 0,
                19: 7,
                20: 8,
                21: 9,
                22: 10,
                23: 11,
                24: 12,
                25: 13,
                26: 14,
                27: 15,
                28: 16,
                29: 0,
                30: 0,
                31: 17,
                32: 18,
                33: 19,
                -1: 0
                }

        label_mask = np.zeros_like(mask)
        for k in mapping_20:
            label_mask[mask == k] = mapping_20[k]
        return label_mask

    def __getitem__(self, index):

        
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])

        
        if self.augment:
          AUG_PROB = 0.5
          # Define data augmentation
          hflip_t = torchvision.transforms.RandomHorizontalFlip(p = 1)
          scale_data = torchvision.transforms.RandomResizedCrop((328, 328), scale=(0.75, 1.0, 1.5, 1.75, 2.0))
          scale_label = torchvision.transforms.RandomResizedCrop((328, 328), scale=(0.75, 1.0, 1.5, 1.75, 2.0))

          aug_pipeline_data = torchvision.transforms.Compose([                                          
                                                torchvision.transforms.RandomApply([hflip_t, scale_data], p = AUG_PROB),
                                                #torchvision.transforms.ToTensor()
                                                ])

          aug_pipeline_label = torchvision.transforms.Compose([                                          
                                                torchvision.transforms.RandomApply([hflip_t, scale_label], p = AUG_PROB),
                                                #torchvision.transforms.ToTensor()
                                                ])
          image = aug_pipeline_data(image)
          label = aug_pipeline_label(label)
        



        name = datafiles["name"]

     

        #print("Initial shape of label: ", np.array(label).shape)
        # resize
        image = image.resize(self.crop_size, Image.BILINEAR)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label)

        label = self.encode_labels(label)

        complete_labels = []
        for c in range(20):
          current_mask = 1 * (label == c)
          complete_labels.append(current_mask)
        
        complete_labels = np.array(complete_labels)

        #print(complete_labels.shape)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        
        
        return image.copy(), complete_labels.copy()


if __name__ == '__main__':
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