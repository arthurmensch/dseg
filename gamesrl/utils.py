import numpy as np


def logsumexp(a, axis=None):
    m = np.max(a, axis=axis)
    a = a - np.expand_dims(m, axis=axis)
    return np.log(np.sum(np.exp(a), axis=axis)) + m


def softmax(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    a = a - m
    r = np.exp(a)
    r /= np.sum(r, axis=axis, keepdims=True)
    return r


class ToIntTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        import torch
        return torch.tensor(np.array(pic)).permute((2, 0, 1))

    def __repr__(self):
        return self.__class__.__name__ + '()'


def weights_init(m, mode='normal'):
    from torch import nn
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        if mode == 'normal':
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0.)
        elif mode == 'kaimingu':
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.)
        elif mode == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, 0.8)