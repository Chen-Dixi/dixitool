import torch
import os
import torch.nn as nn
import shutil


def save_model(root, postfix ,model):
    # type: (str, str, nn.Module) -> void
    """Save a nn.Module to .pth file

    Args:
        root: the root directory to which the model will be save
        postfix: postfix of the save path name
        model: nn.Module
    """
    if not os.path.exists(root):
        os.mkdir(root)

    filename = model.__class__.__name__ + postfix + ".pth"
    PATH = os.path.join(root,filename)
    torch.save(model.state_dict(), PATH)

def save_checkpoint( state, is_best, root, filename='checkpoint.pth.tar'):
    if not os.path.exists(root):
        os.mkdir(root)
    best_name = os.path.join(root, 'best_model_'+filename)
    filename = os.path.join(root, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_name)

def load_from_checkpoint(checkpoint_file, *keys):
    results = []
    checkpoint = torch.load(checkpoint_file)
    for key in keys:
        results.append(checkpoint[key])
    return results

def save_model_dict(root, filename,state_dict):
    # type: (str, str, nn.Module) -> void
    """Save a nn.Module to .pth file

    Args:
        root: the root directory to which the model will be save
        filename: filename
        state_dict: state_dict()
    """
    if not os.path.exists(root):
        os.mkdir(root)

    filename = filename + ".pth"
    PATH = os.path.join(root,filename)
    torch.save(state_dict, PATH)

def load_model_cross_device(PATH, model, save_location='gpu', load_location='cpu'):
    # type: (str, nn.Module, str, str) -> nn.Module
    """Save a nn.Module to .pth file

    Args:
        PATH (str): the PATH of a state_dict file 
        model (nn.Module): model
        save_location (str): where was the model saved before, default value is 'gpu' 
        load_location (str): device location where we load model
    Return:
        nn.Moduel

    """  

    across = save_location+'_'+load_location
    across = across.lower()

    #Save on GPU, Load on CPU
    if across == 'gpu_cpu':
        device = torch.device('cpu')
        model.load_state_dict(torch.load(PATH,map_location=device))
    #Save on GPU, Load on GPU
    elif across == 'gpu_gpu':
        device = torch.device("cuda")
        model.load_state_dict(torch.load(PATH))
        model.to(device)
    #Save on CPU, Load on GPU
    elif across == 'cpu_gpu':
        device = torch.device("cuda")
        model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
        model.to(device)
        # Make sure to call input = input.to(device) on any input tensors that you feed to the model
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(PATH))
        model.to(device)
    return model

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




