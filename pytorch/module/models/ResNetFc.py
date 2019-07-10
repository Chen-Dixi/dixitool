from easydl import *
from torchvision import models


from .BaseFc import BaseFeatureExtractor

class ResNet50Fc(BaseFeatureExtractor):

    def __init__(self, model_path=None,normalize=True ):
        super(ResNet50Fc,self).__init__()

        if model_path:
            self.model_resnet = models.resnet50(pretrained=False)
            self.model_resnet.load_state_dict(torch.load(model_path))
        else:
            self.model_resnet = models.resnet50(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            """This is typically used to register a buffer that should not to be
                considered a model parameter. For example, BatchNorm's ``running_mean``
                is not a parameter, but is part of the persistent state.
                Buffers can be accessed as attributes using given names.
            """
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        #送入全连接层的 dimension
        self.__in_features = model_resnet.fc.in_features

    def forward(self,x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def train(self, mode=True):
        # freeze BN mean and std
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.train(False)
            else:
                module.train(mode)
                
    def output_num(self):
        return self.__in_features








