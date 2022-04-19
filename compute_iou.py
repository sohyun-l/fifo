import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import scipy.misc as m
import warnings
warnings.filterwarnings("ignore")

def fast_hist(a, b, n):
    # import pdb; pdb.set_trace()
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int)+ b[k], minlength=n ** 2).reshape(n, n) #


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def compute_mIoU(gt_dir, pred_dir, devkit_dir, dataset):
    """
    Compute IoU given the predicted colorized images and
    """
    label = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "light",
    "sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motocycle",
    "bicycle"]

    label2train=[
    [0, 255],
    [1, 255],
    [2, 255],
    [3, 255],
    [4, 255],
    [5, 255],
    [6, 255],
    [7, 0],
    [8, 1],
    [9, 255],
    [10, 255],
    [11, 2],
    [12, 3],
    [13, 4],
    [14, 255],
    [15, 255],
    [16, 255],
    [17, 5],
    [18, 255],
    [19, 6],
    [20, 7],
    [21, 8],
    [22, 9],
    [23, 10],
    [24, 11],
    [25, 12],
    [26, 13],
    [27, 14],
    [28, 15],
    [29, 255],
    [30, 255],
    [31, 16],
    [32, 17],
    [33, 18],
    [-1, 255]]    

    num_classes = 19
    name_classes = np.array(label, dtype=np.str)

    hist = np.zeros((num_classes, num_classes))
    if 'FZ' in dataset:
        image_path_list = join(devkit_dir, 'RGB_testv2_filenames.txt')
        label_path_list = join(devkit_dir, 'gt_labelTrainIds_testv2_filenames.txt')
    elif 'FDD' in dataset:
        image_path_list = join(devkit_dir, 'leftImg8bit_testdense_filenames.txt')
        label_path_list = join(devkit_dir, 'gt_testdense_filenames.txt')    
    elif 'FD' in dataset:
        image_path_list = join(devkit_dir, 'leftImg8bit_testall_filenames.txt')
        label_path_list = join(devkit_dir, 'gt_testall_filenames.txt') 
    elif 'Clindau' in dataset:
        image_path_list = join(devkit_dir, 'clear_lindau.txt')
        label_path_list = join(devkit_dir, 'label_lindau.txt')        
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]

    if not 'FZ' in dataset:
        mapping = np.array(label2train, dtype=np.int)
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        
        label = np.array(Image.open(gt_imgs[ind]))
        if not 'FZ' in dataset:
            label = label_mapping(label, mapping)
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

    mIoUs = per_class_iu(hist)
    if 'FZ' in dataset:
        print('Evaluation on Foggy Zurich')
    elif 'FDD' in dataset:
        print('Evaluation on Foggy Driving Dense')
    elif 'FD' in dataset:
        print('Evaluation on Foggy Driving')
    elif 'Clindau' in dataset:        
        print('Evaluation on Cityscapes lindau 40')
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    miou = float(str(round(np.nanmean(mIoUs) * 100, 2)))
    return miou


def miou(args):
   compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir, args.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-dir', type=str, help='directory which stores CityScapes val gt images')
    parser.add_argument('--pred-dir', type=str, help='directory which stores CityScapes val pred images')
    parser.add_argument('--devkit_dir', default='/root/data1/Foggy_Zurich/lists_file_names', help='base directory of zurich')
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    miou(args)
