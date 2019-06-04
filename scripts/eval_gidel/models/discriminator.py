#  MIT License

# Copyright (c) 2017 Ishaan Gulrajani
# Copyright (c) 2017 Marvin Cao
# Copyright (c) Facebook, Inc. and its affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from torch.autograd import Variable, grad
import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self):
        raise NotImplementedError()

    def get_penalty(self, x_true, x_gen):
        x_true = x_true.view_as(x_gen)
        alpha = torch.rand((len(x_true),)+(1,)*(x_true.dim()-1))
        if x_true.is_cuda:
            alpha = alpha.cuda(x_true.get_device())
        x_penalty = Variable(alpha*x_true + (1-alpha)*x_gen, requires_grad=True)
        p_penalty = self.forward(x_penalty)
        gradients = grad(p_penalty, x_penalty, grad_outputs=torch.ones_like(p_penalty).cuda(x_true.get_device())
        if x_true.is_cuda else torch.ones_like(p_penalty), create_graph=True, retain_graph=True, only_inputs=True)[0]
        penalty = ((gradients.view(len(x_true), -1).norm(2, 1) - 1)**2).mean()

        return penalty
