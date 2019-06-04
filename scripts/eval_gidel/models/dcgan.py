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

# written by Hugo Berard (berard.hugo@gmail.com) while at Facebook.

import torch.nn as nn
import torch.nn.functional as F
from .discriminator import Discriminator

class DCGAN32Generator(nn.Module):
    def __init__(self, n_in, n_out, n_filters=128, activation=F.relu, batchnorm=True):
        super(DCGAN32Generator, self).__init__()

        self.n_in = n_in
        self.n_filters = n_filters
        self.activation = activation
        self.batchnorm = batchnorm

        self.deconv1 = nn.Linear(n_in, n_filters*4*4*4)
        self.deconv1_bn = nn.BatchNorm1d(n_filters*4*4*4)
        self.deconv2 = nn.ConvTranspose2d(n_filters*4, n_filters*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(n_filters*2)
        self.deconv3 = nn.ConvTranspose2d(n_filters*2, n_filters, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(n_filters)
        self.deconv5 = nn.ConvTranspose2d(n_filters, n_out, 4, 2, 1)

    def forward(self, z):
        x = self.deconv1(z)
        if self.batchnorm:
            x = self.deconv1_bn(x)
        x = self.activation(x).view(-1,self.n_filters*4,4,4)

        x = self.deconv2(x)
        if self.batchnorm:
            x = self.deconv2_bn(x)
        x = self.activation(x)

        x = self.deconv3(x)
        if self.batchnorm:
            x = self.deconv3_bn(x)
        x = self.activation(x)

        x = F.tanh(self.deconv5(x))

        return x

class DCGAN32Discriminator(Discriminator):
    def __init__(self, n_in, n_out, n_filters=128, activation=F.leaky_relu, batchnorm=True):
        super(DCGAN32Discriminator, self).__init__()

        self.n_filters = n_filters
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv1 = nn.Conv2d(n_in, n_filters, 4, 2, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(n_filters*2)
        self.conv3 = nn.Conv2d(n_filters*2, n_filters*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(n_filters*4)
        self.conv5 = nn.Linear(n_filters*4*4*4, 1)

    def forward(self, x):
        x = self.activation(self.conv1(x))

        x = self.conv2(x)
        if self.batchnorm:
            x = self.conv2_bn(x)
        x = self.activation(x)

        x = self.conv3(x)
        if self.batchnorm:
            x = self.conv3_bn(x)
        x = self.activation(x).view(-1, self.n_filters*4*4*4)

        x = self.conv5(x)

        return x
