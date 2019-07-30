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


class GradientReverseModule(nn.Module):
    """
    wrap GradientReverseLayer to be a nn.Module so that it can be used in ``nn.Sequential``

    usage::

        grl = GradientReverseModule(lambda step : aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

        x = Variable(torch.ones(1), requires_grad=True)
        ans = []
        for _ in range(10000):
            x.grad = None
            y = grl(x)
            y.backward()
            ans.append(variable_to_numpy(x.grad))

        plt.plot(list(range(10000)), ans)
        plt.show() # you can see gradient change from 0 to -1
    """
    def __init__(self, scheduler):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.register_buffer('global_step', torch.zeros(1))
        self.coeff = 0.0
        self.grl = GradientReverseLayer.apply

    def forward(self, x):
        self.coeff = self.scheduler(self.global_step.item())
        if self.training:
            self.global_step += 1.0
        return self.grl(x, self.coeff)
