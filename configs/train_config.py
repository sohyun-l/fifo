import argparse
import numpy as np

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
BETA = 0.005
BATCH_SIZE = 4
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY ='/root/data1'
DATA_LIST_PATH = f'./dataset/cityscapes_list/train_foggy_{BETA}.txt'
DATA_CITY_PATH = './dataset/cityscapes_list/clear_lindau.txt'
INPUT_SIZE = '2048,1024'
DATA_DIRECTORY_CWSF = '/root/data1/Cityscapes'
DATA_LIST_PATH_CWSF = './dataset/cityscapes_list/train_origin.txt'
DATA_LIST_RF = '/root/data1/Foggy_Zurich/lists_file_names/RGB_sum_filenames.txt'
DATA_DIR = '/root/data1'
INPUT_SIZE_RF = '1920,1080'
NUM_CLASSES = 19 
NUM_STEPS = 100000 
NUM_STEPS_STOP = 60000  # early stopping
RANDOM_SEED = 1234
RESTORE_FROM = 'no_model'
RESTORE_FROM_fogpass = 'no_model'
SAVE_PRED_EVERY = 100
SNAPSHOT_DIR = f'/root/data1/snapshots/FIFO_model'   

SET = 'train'

def get_arguments():

    parser = argparse.ArgumentParser(description="FIFO framework")

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY)
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH)
    parser.add_argument("--data-city-list", type=str, default = DATA_CITY_PATH)
    parser.add_argument("--data-list-rf", type=str, default=DATA_LIST_RF)    
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE)
    parser.add_argument("--input-size-rf", type=str, default=INPUT_SIZE_RF)
    parser.add_argument("--data-dir-cwsf", type=str, default=DATA_DIRECTORY_CWSF)
    parser.add_argument("--data-list-cwsf", type=str, default=DATA_LIST_PATH_CWSF)
    parser.add_argument("--data-dir-rf", type=str, default=DATA_DIR)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM)
    parser.add_argument("--restore-from-fogpass", type=str, default=RESTORE_FROM_fogpass)
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY)
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--set", type=str, default=SET)
    parser.add_argument("--lambda-fsm", type=float, default=0.0000001)
    parser.add_argument("--lambda-con", type=float, default=0.0001)
    parser.add_argument("--file-name", type=str, required=True)
    parser.add_argument("--modeltrain", type=str, required=True)
    return parser.parse_args()

args = get_arguments()