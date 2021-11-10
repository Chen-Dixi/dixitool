import argparse
import random
import numpy as np 
import warnings

import torch
import torch.backends.cudnn as cudnn

def print_config(args: argparse.Namespace):
    print("==========================================")
    print("==========       CONFIG      =============")
    print("==========================================")
    for arg, content in args.__dict__.items():
        print("{}:{}".format(arg, content))
    print("\n")

 
def post_config(args: argparse.Namespace):
    args.device = torch.device('cuda:%s'%(args.cuda) if torch.cuda.is_available() else "cpu")
    if args.seed is None:
        args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    cudnn.benchmark = True

    if torch.cuda.is_available() and args.not_cuda:
        warnings.warn("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    return args