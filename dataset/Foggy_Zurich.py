import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
from PIL import Image
from os.path import join
import scipy.misc as m
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class foggyzurichDataSet(data.Dataset):
    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))
    mean_rgb = {
        "cityscapes": [0.0, 0.0, 0.0],
    }
    def __init__(self, root, list_path, max_iters=None, mean=(128, 128, 128), ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(19)))

        for name in self.img_ids:
            img_file = osp.join(self.root, "./Foggy_Zurich/%s" % (name))
            self.files.append({
                "img": img_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        datafiles = self.files[index]
      

        image = Image.open(datafiles["img"]).convert('RGB')
        
        name = datafiles["name"]

        # resize
        w, h = image.size
        image = self._apply_transform(image, scale=0.8)
        
        crop_size = min(600, min(image.size[:2]))
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(crop_size,crop_size)) 
        image = TF.crop(image, i, j, h, w) 

        if random.random() > 0.5:
            image = TF.hflip(image)

        image = np.asarray(image, np.float32)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))


        return image.copy(), np.array(size), name
    
    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def _apply_transform(self, img, scale=(0.7, 1.3), crop_size=600):
        (W, H) = img.size[:2]
        if isinstance(scale, tuple):
            scale = random.random() * 0.6 + 0.7

        tsfrms = []
        tsfrms.append(transforms.Resize((int(H * scale), int(W * scale))))
        tsfrms = transforms.Compose(tsfrms)

        return tsfrms(img)

if __name__ == '__main__':
    dst = foggyzurichDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
