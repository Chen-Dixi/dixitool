import torch
import torchvision.utils as vutils
import os

#Save pytorch model state using the ``.tar`` file extension
def save_model(root,model, epoch,optimizer,loss):
    if not os.path.exists(root):
        raise RuntimeError('Root Directory not found!')

    model_dict = {'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss': loss}

    filename = model.__class__.__name__ + ("_epoch_%d" % epoch) + ".tar"
    print("Saving to ",filename)
    PATH = os.path.join(root, filename)
    torch.save(model_dict, PATH)

def save_model_1(root,model,loss,accuracy):
    if not os.path.exists(root):
        raise RuntimeError('Root Directory not found!')

    model_dict = {'model_state_dict': model.state_dict(),
            'accuracy': accuracy,
            'loss': loss}

    filename = model.__class__.__name__ + ("_acc_%.2f" % accuracy) + ".tar"
    print("Saving to ",filename)
    PATH = os.path.join(root, filename)
    torch.save(model_dict, PATH)

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
    filename = filename+".png"
    PATH = os.path.join(root,filename)
    tensor = torch.from_numpy(data)
    vutils.save_image(tensor, PATH,padding=2,nrow=nrow, normalize=True)