import os
import os.path as osp

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data

import argparse
import numpy as np
from packaging import version
import wandb
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from model.refinenetlw import rf_lw101
from compute_iou import compute_mIoU
from configs.test_config import get_arguments
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.Foggy_Zurich_test import foggyzurichDataSet
from dataset.foggy_driving import foggydrivingDataSet

RESTORE_FROM = 'without_pretraining'
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def eval():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    if args.restore_from == RESTORE_FROM:
        start_iter = 0
        model = rf_lw101(num_classes=args.num_classes)

    else:
        restore = torch.load(args.restore_from)
        model = rf_lw101(num_classes=args.num_classes)

        model.load_state_dict(restore['state_dict'])
        start_iter = 0

    save_dir_fz = osp.join(f'./result_FZ', args.file_name)
    save_dir_fd = osp.join(f'./result_FD', args.file_name)
    save_dir_fdd = osp.join(f'./result_FDD', args.file_name)
    save_dir_clindau = osp.join(f'./result_Clindau', args.file_name)      

    if not os.path.exists(save_dir_fz):
        os.makedirs(save_dir_fz)
    if not os.path.exists(save_dir_fd):
        os.makedirs(save_dir_fd)
    if not os.path.exists(save_dir_fdd):
        os.makedirs(save_dir_fdd)
    if not os.path.exists(save_dir_clindau):
        os.makedirs(save_dir_clindau)
    
    model.eval()
    device = torch.device('cuda:0')
    model.to(device)

    testloader1 = data.DataLoader(foggyzurichDataSet(args.data_dir_eval, args.data_list_eval, crop_size=(1152, 648), mean=IMG_MEAN),
                                    batch_size=1, shuffle=False, pin_memory=True)
    testloader2 = data.DataLoader(foggyzurichDataSet(args.data_dir_eval, args.data_list_eval, crop_size=(1536, 864), mean=IMG_MEAN),
                                    batch_size=1, shuffle=False, pin_memory=True)
    testloader3 = data.DataLoader(foggyzurichDataSet(args.data_dir_eval, args.data_list_eval, crop_size=(1920, 1080), mean=IMG_MEAN),
                                    batch_size=1, shuffle=False, pin_memory=True)

    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp_eval = nn.Upsample(size=(1080,1920), mode='bilinear', align_corners=True)
    else:
        interp_eval = nn.Upsample(size=(1080,1920), mode='bilinear')

    testloader_iter2 = enumerate(testloader2)
    testloader_iter3 = enumerate(testloader3)


    for index, batch1 in enumerate(testloader1):
        image, label_test, _, name = batch1
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
            output_1 = interp_eval(output2)

        _, batch2 = testloader_iter2.__next__()
        image, label_test, _, name = batch2
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
            output_2 = interp_eval(output2)

        _, batch3 = testloader_iter3.__next__()    
        image, label_test, _, name = batch3
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
            output_3 = interp_eval(output2)

        output = torch.cat([output_1,output_2,output_3])
        output = torch.mean(output, dim=0)
        output = output.cpu().numpy()
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        name = name[0].split('/')[-1]
        output.save('%s/%s' % (save_dir_fz, name))
        output_col.save('%s/%s_color.png' % (save_dir_fz, name[:-4]))
    miou_fz = compute_mIoU(args.gt_dir_fz, save_dir_fz, args.devkit_dir_fz, 'FZ')


    testloader1 = data.DataLoader(foggydrivingDataSet(args.data_dir_eval_fd, args.data_list_eval_fdd, scale=1),
                                    batch_size=1, shuffle=False, pin_memory=True)

    testloader2 = data.DataLoader(foggydrivingDataSet(args.data_dir_eval_fd, args.data_list_eval_fdd, scale=0.8),
                                    batch_size=1, shuffle=False, pin_memory=True) 

    testloader3 = data.DataLoader(foggydrivingDataSet(args.data_dir_eval_fd, args.data_list_eval_fdd, scale=0.6),
                                    batch_size=1, shuffle=False, pin_memory=True)
    testloader_iter2 = enumerate(testloader2)
    testloader_iter3 = enumerate(testloader3)

    for index, batch in enumerate(testloader1):
        image, size, name = batch
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
            interp_eval = nn.Upsample(size=(size[0][0],size[0][1]), mode='bilinear')
            output_1 = interp_eval(output2)

        _, batch2 = testloader_iter2.__next__()
        image, _, name = batch2
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
            output_2 = interp_eval(output2)

        _, batch3 = testloader_iter3.__next__()    
        image, _, name = batch3
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
            output_3 = interp_eval(output2)

        output = torch.cat([output_1,output_2,output_3])
        output = torch.mean(output, dim=0)
        output = output.cpu().numpy()
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        name = name[0].split('/')[-1]
        output.save('%s/%s' % (save_dir_fdd, name))
        output_col.save('%s/%s_color.png' % (save_dir_fdd, name[:-4]))
    miou_fdd = compute_mIoU(args.gt_dir_fd, save_dir_fdd, args.devkit_dir_fd, 'FDD')


    testloader1 = data.DataLoader(foggydrivingDataSet(args.data_dir_eval_fd, args.data_list_eval_fd, scale=1),
                                    batch_size=1, shuffle=False, pin_memory=True) 

    testloader2 = data.DataLoader(foggydrivingDataSet(args.data_dir_eval_fd, args.data_list_eval_fd, scale=0.8),
                                    batch_size=1, shuffle=False, pin_memory=True) 

    testloader3 = data.DataLoader(foggydrivingDataSet(args.data_dir_eval_fd, args.data_list_eval_fd, scale=0.6),
                                    batch_size=1, shuffle=False, pin_memory=True) 
    testloader_iter2 = enumerate(testloader2)
    testloader_iter3 = enumerate(testloader3)

    for index, batch in enumerate(testloader1):
        image, size, name = batch
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
            interp_eval = nn.Upsample(size=(size[0][0],size[0][1]), mode='bilinear')

            output_1 = interp_eval(output2)

        _, batch2 = testloader_iter2.__next__()
        image, _, name = batch2
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
            output_2 = interp_eval(output2)

        _, batch3 = testloader_iter3.__next__()    
        image, _, name = batch3
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
            output_3 = interp_eval(output2)

        output = torch.cat([output_1,output_2,output_3])
        output = torch.mean(output, dim=0)
        output = output.cpu().numpy()
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        name = name[0].split('/')[-1]
        output.save('%s/%s' % (save_dir_fd, name))
        output_col.save('%s/%s_color.png' % (save_dir_fd, name[:-4]))
    miou_fd = compute_mIoU(args.gt_dir_fd, save_dir_fd, args.devkit_dir_fd, 'FD')


    testloader1 = data.DataLoader(cityscapesDataSet(args.data_dir_city, args.data_city_list, crop_size = (2048, 1024), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                            batch_size=1, shuffle=False, pin_memory=True)
    testloader2 = data.DataLoader(cityscapesDataSet(args.data_dir_city, args.data_city_list, crop_size = (2048*0.8, 1024*0.8), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                            batch_size=1, shuffle=False, pin_memory=True)
    testloader3 = data.DataLoader(cityscapesDataSet(args.data_dir_city, args.data_city_list, crop_size = (2048*0.6, 1024*0.6), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                            batch_size=1, shuffle=False, pin_memory=True)   
    testloader_iter2 = enumerate(testloader2)
    testloader_iter3 = enumerate(testloader3)

    for index, batch in enumerate(testloader1):
        image, size, name = batch
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
            interp_eval = nn.Upsample(size=(1024, 2048), mode='bilinear')
            output_1 = interp_eval(output2)

        _, batch2 = testloader_iter2.__next__()
        image, _, name = batch2
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
            output_2 = interp_eval(output2)

        _, batch3 = testloader_iter3.__next__()    
        image, _, name = batch3
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
            output_3 = interp_eval(output2)

        output = torch.cat([output_1,output_2,output_3])
        output = torch.mean(output, dim=0)
        output = output.cpu().numpy()
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        name = name[0].split('/')[-1]
        output.save('%s/%s' % (save_dir_clindau, name))
        output_col.save('%s/%s_color.png' % (save_dir_clindau, name.split('.')[0]))

    miou_clindau = compute_mIoU(args.gt_dir_clindau, save_dir_clindau, args.devkit_dir_clindau, 'Clindau')



if __name__ == '__main__':
    eval()
