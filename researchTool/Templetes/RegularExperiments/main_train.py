import _ProjectName_.functions as functions
from config import get_train_arguments
from training import *
import random
#都可以用的就放里面，在另外定制的加外面
parser = get_train_arguments()
#get_train_arguments里面就放必要的arguments
#接下来 parser里面可以加一些别的东西
#parser.add_argument('--checkpoint-file-name', default='step2checkpoint.pth.tar', type=str, metavar='checkpoint-file-name',
#  help='name for checkpoint files') #只给文件名，一般后面的处理都会在path前面加一个checkpoints文件夹

#设置，所以叫option
opt = parser.parse_args()

opt=functions.post_config(opt) #给args加一些 argparse加不了的东西
functions.print_config(opt) #打印 parser.parse_args()里面的东西

#先生成要保存的地址
if (os.path.exists(opt.dir2save)):
    print('trained model already exist')
else:
    try:
        os.makedirs(opt.dir2save)
    except OSError:
        pass
 

#找 model

#记载数据集

#accMetric



