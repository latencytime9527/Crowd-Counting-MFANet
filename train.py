from utils.regression_trainer import  MFANetTrainer
import argparse
import os
import torch
args = None
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--data-dir', default='/home/sda_disk/sda_disk/data/ShanghaiTech_Dataset/part_B_final/')
    parser.add_argument('--save-dir', default='./checkpoint')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--resume', default='')
    #parser.add_argument('--max-model-num', type=int, default=1)
    parser.add_argument('--max-epoch', type=int, default=400)
    parser.add_argument('--val-epoch', type=int, default=1)
    parser.add_argument('--val-start', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', type=int, default=1 )
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--is-gray', type=bool, default=False)
    parser.add_argument('--downsample-ratio', type=int, default=8)
    parser.add_argument('--seed', default = time.time())
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = MFANetTrainer(args)
    trainer.setup()
    trainer.train()
