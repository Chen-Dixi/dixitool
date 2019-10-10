# Function should be defined in a different way in the newer versions of pytorch:
from torch.autograd import Function
import torch
import torch.nn as nn

class GradReverse(Function):

    #pytorch 1.0之后都只能在
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.save_for_backward(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambd = ctx.saved_tensors[0]
        return grad_output.neg()*lambd, None

def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)

# Use grl as a module
def identity_scheduler(global_step):
    return torch.ones(1)

#lambda 在这一层里面计算
class GradReverseLayer(nn.Module):


    def __init__(self, scheduler=identity_scheduler):
        super(GradReverseLayer, self).__init__()
        self.scheduler = scheduler
        self.register_buffer('global_step', torch.zeros(1))


    def forward(self, x):
        #1个batch就加一个1
        lambd = self.scheduler(self.global_step)#lambd是一个torch
        if self.training:
            self.global_step += 1.0
        x = grad_reverse(x, lambd)
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
