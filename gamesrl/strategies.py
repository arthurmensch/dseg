import torch
import torch.nn as nn
from torch.nn import Parameter


class BypassedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, dim):
        return torch.softmax(f, dim)

    @staticmethod
    def backward(ctx, grad_f):
        return grad_f.view_as(grad_f), None


class MatrixGameStrategy(nn.Module):
    def __init__(self, n_actions, mirror=True):
        super().__init__()
        self.mirror = mirror
        self.strategy = Parameter(torch.empty(n_actions))
        self.reset_parameters()

    def reset_parameters(self):
        self.strategy.data.normal_(0, 1)

    def forward(self):
        if self.mirror:
            return BypassedSoftmax.apply(self.strategy, 0)
        else:
            return torch.softmax(self.strategy, 0)


class IteratedGameStrategy(nn.Module):
    def __init__(self, mirror=True):
        super().__init__()
        self.strategy = Parameter(torch.empty(2, 5))
        self.mirror = mirror

    def reset_parameters(self):
        self.strategy.data.normal(0, 1)

    def forward(self):
        if self.mirror:
            return BypassedSoftmax.apply(self.strategy, 0)[0]
        else:
            return torch.softmax(self.strategy, 0)[0]
