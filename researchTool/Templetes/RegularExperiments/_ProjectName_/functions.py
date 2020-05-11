import torch
import random
import numpy as np 

def print_config(args):
    print("==========================================")
    print("==========       CONFIG      =============")
    print("==========================================")
    for arg, content in args.__dict__.items():
        print("{}:{}".format(arg, content))
    print("\n")

 
def post_config(opt):
    opt.device = torch.device('cuda:%s'%(opt.cuda) if torch.cuda.is_available() else "cpu")
    if opt.seed is None:
        opt.seed = random.randint(1, 10000)
    
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    #输出文件
    opt.dir2save = generate_dir2save(opt)
    return opt

def generate_dir2save(opt):
    dir2save = "TrainedModel/%s/%d" % (opt.net, opt.build)
    return dir2save


import logging

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger