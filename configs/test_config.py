
import argparse
import numpy as np

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
MODEL = 'RefineNetNew'
DATA_DIRECTORY ='/root/data1'
DATA_CITY_PATH = './dataset/cityscapes_list/clear_lindau.txt'
DATA_DIRECTORY_CITY = '/root/data1/Cityscapes'
DATA_LIST_PATH_EVAL = '/root/data1/Foggy_Zurich/lists_file_names/RGB_testv2_filenames.txt'
DATA_LIST_PATH_EVAL_FD ='./lists_file_names/leftImg8bit_testall_filenames.txt'
DATA_LIST_PATH_EVAL_FDD ='./lists_file_names/leftImg8bit_testdense_filenames.txt' 
DATA_DIR_EVAL = '/root/data1'
DATA_DIR_EVAL_FD = '/root/data1/Foggy_Driving'
NUM_CLASSES = 19 
RESTORE_FROM = 'no model'
SNAPSHOT_DIR = f'/root/data1/snapshots/FIFO'
GT_DIR_FZ = '/root/data1/Foggy_Zurich'
GT_DIR_FD = '/root/data1/Foggy_Driving'
GT_DIR_CLINDAU = '/root/data1/Cityscapes/gtFine'
SET = 'val'

MODEL = 'RefineNetNew'

def get_arguments():
    parser = argparse.ArgumentParser(description="Evlauation")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY)
    parser.add_argument("--data-city-list", type=str, default = DATA_CITY_PATH)
    parser.add_argument("--data-list-eval-fd", type=str, default=DATA_LIST_PATH_EVAL_FD)      
    parser.add_argument("--data-list-eval-fdd", type=str, default=DATA_LIST_PATH_EVAL_FDD)             
    parser.add_argument("--data-dir-city", type=str, default=DATA_DIRECTORY_CITY)
    parser.add_argument("--data-list-eval", type=str, default=DATA_LIST_PATH_EVAL)
    parser.add_argument("--data-dir-eval", type=str, default=DATA_DIR_EVAL)
    parser.add_argument("--data-dir-eval-fd", type=str, default=DATA_DIR_EVAL_FD)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM)    
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--set", type=str, default=SET)
    parser.add_argument("--file-name", type=str, required=True)
    parser.add_argument("--gt-dir-fz", type=str, default=GT_DIR_FZ)
    parser.add_argument("--gt-dir-fd", type=str, default=GT_DIR_FD)
    parser.add_argument("--gt-dir-clindau", type=str, default=GT_DIR_CLINDAU)
    parser.add_argument("--devkit-dir-fz", default='/root/data1/Foggy_Zurich/lists_file_names') 
    parser.add_argument("--devkit-dir-fd", default='./lists_file_names') 
    parser.add_argument("--devkit-dir-clindau", default='./dataset/cityscapes_list')
    return parser.parse_args()
