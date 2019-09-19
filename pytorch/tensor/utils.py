import os
import torch
import torchvision.utils as vutils

#Save an numpy object of an image to a ``png`` file 
def save_image(root, tensor, epoch , iters):
    if not os.path.exists(root):
        raise RuntimeError('Root Directory not found!')
    filename = "fake_image"+ ("_epoch_%d_iters_%d" % (epoch,iters) ) + ".png"
    PATH = os.path.join(root,filename)
    vutils.save_image(tensor, PATH,padding=2, normalize=True)

def save_tensor2img(root,filename,tensor,padding=2, nrow=8):
    if not os.path.exists(root):
        raise RuntimeError('Root Directory not found!')
    filename = filename + ".png"
    PATH = os.path.join(root,filename)
    vutils.save_image(tensor, PATH,padding=padding, nrow=nrow,normalize=True)

def save_image_from_numpy(root,filename,data,nrow=8):
    if not os.path.exists(root):
        raise RuntimeError('Root Directory not found!')
    if not any((filename.lower()).endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
        filename = filename+".png"
    PATH = os.path.join(root,filename)
    tensor = torch.from_numpy(data)
    vutils.save_image(tensor, PATH,padding=2,nrow=nrow, normalize=True)