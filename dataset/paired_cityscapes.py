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

class Pairedcityscapes(data.Dataset):
    colors = [  
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

    def __init__(self, src_root, trg_root, src_list_path, trg_list_path, max_iters=None, mean=(128, 128, 128), ignore_label=255, set='val'):
        self.src_root = src_root
        self.trg_root = trg_root
        self.src_list_path = src_list_path
        self.trg_list_path = trg_list_path
        self.ignore_label = ignore_label
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(src_list_path)]
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
            trg_img_file = osp.join(self.trg_root, "leftImg8bit/%s/%s" % (self.set, name[:-21]+'.png'))
            src_img_file = osp.join(self.src_root, "./leftImg8bit_foggyDBF/%s/%s" % (self.set, name))
            label_file = osp.join(self.src_root, "./Cityscapes/gtFine/%s/%s" % (self.set, name[:-32]+'gtFine_labelIds.png'))
            self.files.append({
                "src_img": src_img_file,
                "trg_img": trg_img_file,
                "label": label_file,
                "trg_name": name[:-21]+'.png',
                "src_name": name
            })



    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        datafiles = self.files[index]
      
        src_image = Image.open(datafiles["src_img"]).convert('RGB')
        trg_image = Image.open(datafiles["trg_img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        src_name = datafiles["src_name"]
        trg_name = datafiles["trg_name"]
        # resize
        w, h = src_image.size
        src_image, trg_image, label = self._apply_transform(src_image, trg_image, label, scale=0.8)

        crop_size = min(600, min(src_image.size[:2]))
        i, j, h, w = transforms.RandomCrop.get_params(src_image, output_size=(crop_size,crop_size)) 
        src_image = TF.crop(src_image, i, j, h, w) 
        trg_image = TF.crop(trg_image, i, j, h, w) 
        label = TF.crop(label, i, j, h, w)

        if random.random() > 0.5:
            src_image = TF.hflip(src_image)
            trg_image = TF.hflip(trg_image)
            label = TF.hflip(label)

        src_image = np.asarray(src_image, np.float32)
        trg_image = np.asarray(trg_image, np.float32)
        label = self.encode_segmap(np.array(label, dtype=np.float32))

        classes = np.unique(label)
        lbl = label.astype(float)

        label = lbl.astype(int)

        size = src_image.shape
        src_image = src_image[:, :, ::-1]  # change to BGR
        src_image -= self.mean
        src_image = src_image.transpose((2, 0, 1))
        trg_image = trg_image[:, :, ::-1]  # change to BGR
        trg_image -= self.mean
        trg_image = trg_image.transpose((2, 0, 1))


        return src_image.copy(), trg_image.copy(), label.copy(), np.array(size), src_name, trg_name
    
    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def _apply_transform(self, img1, img2, lbl, scale=(0.7, 1.3), crop_size=600):
        (W, H) = img1.size[:2]
        if isinstance(scale, tuple):
            scale = random.random() * 0.6 + 0.7

        tsfrms = []
        tsfrms.append(transforms.Resize((int(H * scale), int(W * scale))))
        tsfrms = transforms.Compose(tsfrms)

        return tsfrms(img1), tsfrms(img2), tsfrms(lbl)

