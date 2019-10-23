# Function should be defined in a different way in the newer versions of pytorch:
from torch.autograd import Function
import torch
import torch.nn as nn

#这个代码有很大的gpu tensor 和cpu tensor不能一起计算的问题
class GradReverse(Function):

    #pytorch 1.0之后都只能在
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambd = ctx.lambd
        return grad_output.neg()*lambd, None

def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)

# Use grl as a module
def identity_scheduler(global_step):
    return torch.ones(1).item()

#lambda 在这一层里面计算
class GradReverseLayer(nn.Module):


    def __init__(self, scheduler=identity_scheduler):
        super(GradReverseLayer, self).__init__()
        self.scheduler = scheduler
        self.register_buffer('global_step', torch.zeros(1))

    #注意这里的
    def forward(self, x , lambd=None):
        #1个batch就加一个1
        if lambd is None:
            lambd = self.scheduler(self.global_step.item())#lambd是一个torch tensor
        
        if self.training:
            self.global_step += 1.0
        x = GradReverse.apply(x, lambd)
        return x


###============测试
# def test():
#     model = GradReverseLayer()
#     sample = torch.ones([2,1],requires_grad=True)
#     y = model(sample)
#     y = (y*2).sum()
#     y.backward()
#     print(sample.grad)

# test()
