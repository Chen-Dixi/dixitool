from .BaseFc import BaseFeatureExtractor
from torchvision import models

class VGG16Fc(BaseFeatureExtractor):
    def __init__(self,model_path=None, normalize=True):
        super(VGG16Fc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_vgg = models.vgg16(pretrained=False)
                self.model_vgg.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            self.model_vgg = models.vgg16(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        model_vgg = self.model_vgg
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
        self.feature_layers = nn.Sequential(self.features, self.classifier)

        self.__in_features = 4096

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.features(x)
        x = x.view(x.size(0), 25088)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features