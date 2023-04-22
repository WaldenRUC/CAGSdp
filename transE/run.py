import argparse, random, pickle, torch, os
import numpy as np
from file_dataset import TransEDataset
from model import *
def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
parser = argparse.ArgumentParser()
parser.add_argument('--epochs',     default=50000,  type=int)
parser.add_argument('--batch_size', default=128,    type=int)
parser.add_argument('--seed', default=0,    type=int)
args = parser.parse_args()
myDataset = TransEDataset(datapath="/home/zhaoheng_huang/CAGS_data/TransE/train.txt")
if __name__ == "__main__":
    set_seed(args.seed)
    