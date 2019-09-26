import torch.nn as nn 

class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

    def _init_params(self):
        for m in self.modules:
            if isinstance(m ,nn.Conv2d)
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinear='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)