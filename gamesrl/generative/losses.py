#  MIT License

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


import torch
from torch.autograd import grad
from torch.nn import functional as F, Parameter


def compute_gan_loss(true_logits, fake_logits, loss_type='gan', compute_D=True, compute_G=True):
    """Return the generator loss"""
    if loss_type in ['gan', 'ns-gan']:
        if compute_G:
            if loss_type == 'ns-gan':
                loss_G = - F.logsigmoid(fake_logits).mean()
            else:
                loss_G = - F.softplus(fake_logits).mean()  # + F.logsigmoid(true_logits).mean()
        else:
            loss_G = None
        if compute_D:
            true_loss_D = loss_D = - F.logsigmoid(true_logits).mean()
            if loss_type != 'ns-gan':
                true_loss_D = true_loss_D.item()
            else:
                true_loss_D = 0
            if loss_type == 'gan' and compute_G:  # reuse previously computed loss_G
                loss_D -= loss_G
            else:
                loss_D += F.softplus(fake_logits).mean()
        else:
            loss_D = None
            true_loss_D = None
    elif loss_type in ['wgan', 'wgan-gp']:
        if compute_G:
            loss_G = - fake_logits.mean()   # + true_logits.mean()
        else:
            loss_G = None
        if compute_D:
            true_loss_D = loss_D = - true_logits.mean()
            true_loss_D = true_loss_D.item()
            if compute_G:
                loss_D -= loss_G
            else:
                loss_D += fake_logits.mean()
        else:
            loss_D = None
            true_loss_D = None
    else:
        raise NotImplementedError()

    return loss_D, loss_G, true_loss_D


def compute_grad_penalty(net_D, true_data, fake_data):
    batch_size = true_data.shape[0]
    epsilon = true_data.new(batch_size, 1, 1, 1)
    epsilon = epsilon.uniform_()
    line_data = true_data * (1 - epsilon) + fake_data * (1 - epsilon)
    line_data = Parameter(line_data)
    line_pred = net_D(line_data).sum()
    grad, = torch.autograd.grad(line_pred, line_data, create_graph=True)
    grad = grad.view(batch_size, -1)
    grad_norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    return ((grad_norm - 1) ** 2).mean()