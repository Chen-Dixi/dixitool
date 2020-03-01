import argparse

def get_train_arguments():
    parser = argparse.ArgumentParser(description="Open Set Larg Margin Gaussian")
    
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--checkpoint-file-name', default='step2checkpoint.pth.tar', type=str,help='name for checkpoint files') #只给文件名，一般后面的处理都会在path前面加一个checkpoints文件夹
    parser.add_argument('--net', type=str, default='alex', metavar='B',
                        help='which network alex,vgg,res?')
    parser.add_argument('--imagenet-pretrained',type=bool, default=True,metavar='N',help='imagenet pretrained option(default: True)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--test-interval', type=int, default=4, metavar='N',
                        help='how many epochs to wait before test')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 128)')
    #parser.add_argument('--resume', default=False, type=bool, metavar='PATH',
                        #help='path to latest checkpoint (default: none)')
    
    parser.add_argument('--seed', type=int, default=19950907, metavar='S',
                        help='random seed (default: 19950907) my birthday')
    parser.add_argument('--result',type=str, default='result',help='directory to save txt results')
    parser.add_argument('--cuda',type=int,default=0)
    
    parser.add_argument('--best-criterion',type=str,default='ALL',metavar='best_cri',choices=['ALL','OS','OS*','UNK'])
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
    parser.add_argument('--build',type=int,default=0,help="the number of build")
    return parser

def get_inference_arguments():
    arser = argparse.ArgumentParser(description="")
    parser.add_argument('--cuda',type=int,default=0,metavar='cuda')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--load-path',default='checkpoints/best_model_dslr_amazon_step2checkpoint.pth.tar',type=str)
    parser.add_argument('--figure-name',default='likelihood',type=str)
    parser.add_argument('--domain',default='amazon',type=str)
    parser.add_argument('--source',default='dslr',type=str)
    parser.add_argument('--domain-path',default='datasets/target_amazon_caltech_share.txt',type=str)
    parser.add_argument('--imagenet-pretrained',type=bool, default=True,metavar='N',help='imagenet pretrained option(default: True)')
    parser.add_argument('--seed', type=int, default=19950907, metavar='S',
                        help='random seed (default: 19950907) my birthday')
    parser.add_argument('--net', type=str, default='vgg19', metavar='B',
                        help='which network alex,vgg,res?')

    return parser