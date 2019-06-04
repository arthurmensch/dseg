from .dcgan import DiscriminatorDCGAN32, DiscriminatorDCGAN64, GeneratorDCGAN32, GeneratorDCGAN64
from .resnet import DiscriminatorResNet32, GeneratorResNet32
from gamesrl.generative.losses import compute_gan_loss, compute_grad_penalty
from .validation import InceptionScorer