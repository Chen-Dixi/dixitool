import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import os

def create_data_loader(dataset, root, batch_size):
    
    if dataset.lower() == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
            ])

        #Prepa Data  ,Train, Test分开
        #mnist
        trainSets = datasets.MNIST(root,train=True,download=False,transform=transform)
        trainDataloader = DataLoader(dataset=trainSets,batch_size=batch_size,shuffle=True)

        testSets = datasets.MNIST(root,train=False,download=False,transform=transform)
        testDataloader = DataLoader(dataset=testSets,batch_size=batch_size,shuffle=True)
        
        
    elif dataset.lower() == "mnist_m":
        dataset_mean = (0.5, 0.5, 0.5)
        dataset_std = (0.5, 0.5, 0.5)
        transform = transforms.Compose([
                transforms.Resize([28,28]),
                transforms.ToTensor(),
                transforms.Normalize(mean=dataset_mean,std=dataset_std)
            ])

        #mnist_m
        trainSets = datasets.ImageFolder(root=os.path.join(root,'train'),transform=transform)
        trainDataloader = DataLoader(dataset=trainSets,batch_size=batch_size,shuffle=True)

        testSets = datasets.ImageFolder(root=os.path.join(root,'test'),transform=transform)
        

        testDataloader = DataLoader(dataset=testSets,batch_size=batch_size,shuffle=True)
    return trainDataloader, testDataloader

def create_data_loader_with_transform(dataset, root, batch_size, train_transform=None,test_transform=None):
        # type: (str,str,int,tranform, transform) -> dataloader
    """create various dataloader object

    Args:
        str,str,tranform, transform
    Return:
    """
    if dataset.lower() == "mnist":
        #Prepa Data  ,Train, Test分开
        #mnist
        trainSets = datasets.MNIST(root,train=True,download=False,transform=training_transform)
        trainDataloader = DataLoader(dataset=trainSets,batch_size=batch_size,shuffle=True)

        testSets = datasets.MNIST(root,train=False,download=False,transform=test_transform)
        testDataloader = DataLoader(dataset=testSets,batch_size=batch_size,shuffle=True)
        return trainDataloader, testDataloader
        
    elif dataset.lower() == "mnist_m":
        #mnist_m
        trainSets = datasets.ImageFolder(root=os.path.join(root,'train'),transform=train_transform)
        trainDataloader = DataLoader(dataset=trainSets,batch_size=batch_size,shuffle=True)

        testSets = datasets.ImageFolder(root=os.path.join(root,'test'),transform=test_transform)
        
        testDataloader = DataLoader(dataset=testSets,batch_size=batch_size,shuffle=True)

        return trainDataloader, testDataloader
    elif dataset.lower() == "webcam" or dataset.lower() == "dslr" or dataset.lower() == "amazon":
        #office-31 datasets 主要的问题是划分 test和train
        # office-31 数据集，不反回测试集
        trainSets = datasets.ImageFolder(root=os.path.join(root,'images'), transform=train_transform)
        dataloader = DataLoader(dataset=trainSets,batch_size=batch_size,shuffle=True)

        return dataloader
    else:
        raise ValueError("dataset cannot be recognized")


    






