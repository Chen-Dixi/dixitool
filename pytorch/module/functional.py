import torch
import os




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






