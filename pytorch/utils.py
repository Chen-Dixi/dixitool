import torch
import torch.nn as nn

class GradientReverseLayer(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
         # this is necessary. if we just return ``input``, ``backward`` will not be called sometimes
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        grad_output = grad_outputs.neg() * coeff
        return grad_output, None


