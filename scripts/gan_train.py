import copy
import math
import os
import random
import time
from os.path import expanduser, join

import numpy as np
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
# Hack to test
from sacred import Experiment
from sacred.observers import FileStorageObserver
from torch.optim import SGD, RMSprop, Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import TensorDataset, DataLoader

from gamesrl.generative import (DiscriminatorDCGAN64, GeneratorDCGAN64, DiscriminatorDCGAN32,
                                GeneratorDCGAN32, compute_gan_loss, compute_grad_penalty, GeneratorResNet32,
                                DiscriminatorResNet32)
from gamesrl.generative.scores import compute_scores, compute_activations
from gamesrl.optimizers import ExtraOptimizer, ExtraOptimizerVR

exp = Experiment('gan')
exp_dir = expanduser('~/output/games_rl/gan_clean')
data_dir = expanduser('~/data/games_rl')
stat_dir = expanduser('~/data/games_rl/stats')

if not os.path.exists(stat_dir):
    os.makedirs(stat_dir)


def infinite_iter(generator):
    while True:
        for e in generator:
            yield e


@exp.capture
def generate(noise, net_G, batch_size, device):
    test_samples = noise.shape[0]
    dataset = TensorDataset(noise)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    images = np.empty((test_samples, 3, 32, 32), dtype=np.uint8)

    start, end = 0, 0
    for i, (this_noise,) in enumerate(dataloader):
        this_noise = this_noise.to(device)
        end = start + len(this_noise)
        with torch.no_grad():
            these_images = net_G(this_noise)
        images[start:end] = these_images.add(1).mul_(255 / 2).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
        start = end
    return images


def update_avg(net, avg, weight):
    for net_param, avg_param in zip(net.parameters(), avg.parameters()):
        avg_param.data *= (1 - weight)
        avg_param.data += weight * net_param.data


def save_checkpoint(net_G, net_D, avg_net_G, optimizer_G, optimizer_D, scheduler_G, scheduler_D, path):
    checkpoint = {'net_G': net_G.state_dict(),
                  'net_D': net_D.state_dict(),
                  'avg_net_G': avg_net_G.state_dict(),
                  'optimizer_D': optimizer_D.state_dict(),
                  'optimizer_G': optimizer_G.state_dict(),
                  'scheduler_D': scheduler_D.state_dict(),
                  'scheduler_G': scheduler_G.state_dict(),
                  }
    torch.save(checkpoint, path)


def clip_weights(net, clip_value=1.):
    for param in net.parameters():
        param.data.clamp_(min=-clip_value, max=clip_value)


def compute_grad_norm(model):
    return sum(torch.sum(param.grad ** 2) for param in model.parameters())


def load_checkpoint(net_G, net_D, avg_net_G, optimizer_G, optimizer_D, scheduler_G, scheduler_D, path):
    checkpoint = torch.load(path)
    net_G.load_state_dict(checkpoint['net_G'])
    avg_net_G.load_state_dict(checkpoint['avg_net_G'])
    net_D.load_state_dict(checkpoint['net_D'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D'])
    try:
        scheduler_D.load_state_dict(checkpoint['scheduler_D'])
        scheduler_G.load_state_dict(checkpoint['scheduler_G'])
    except KeyError:
        pass


@exp.config
def config():
    seed = 1234
    n_jobs = 4
    n_iter = int(5e5)

    dataset = 'lsun'  # cifar10 | lsun | mnist |imagenet | folder | lfw | fake
    device = 'cuda:0'
    test_device = 'cuda:0'

    test_samples = 50000

    batch_size = 64
    image_size = 32

    architecture = 'resnet'
    ngf = 128
    ndf = 128
    nz = 128
    D_batch_norm = False

    checkpoint = None

    loss_type = 'wgan-gp'
    grad_penalty = 10  # For wgan-gp only
    clip_value = 0.01

    optimizer = 'adam'
    lr = 5e-4
    G_lr_factor = .1
    beta1 = 0.5
    beta2 = 0.9
    lr_decay = False

    D_importance = 1
    G_importance = 1

    sampling = 'alternated'
    extrapolation = True
    variance_reduction = False
    importance_sampling = False
    compute_is = True
    compute_fid = True

    print_every = 500
    record_every = 10000


@exp.named_config
def extragradient_vr():
    lr = 5e-4
    G_lr_factor = .1
    sampling = 'one'
    extrapolation = True
    importance_sampling = False
    variance_reduction = True


@exp.named_config
def extragradient_alternated():
    lr = 5e-4
    G_lr_factor = .1
    sampling = 'alternated'
    extrapolation = True
    importance_sampling = False
    variance_reduction = False


@exp.named_config
def extragradient_sampled():
    lr = 5e-4
    G_lr_factor = .1
    sampling = 'one'
    extrapolation = True
    importance_sampling = False
    variance_reduction = False


@exp.named_config
def extragradient():
    lr = 5e-4
    G_lr_factor = .1
    sampling = 'all'
    extrapolation = True
    importance_sampling = False
    variance_reduction = False


@exp.named_config
def alternated():
    lr = 5e-4
    G_lr_factor = .1
    extrapolation = False
    sampling = 'alternated'

@exp.named_config
def simultaneous():
    extrapolation = False
    sampling = 'all'


@exp.capture
def make_dataset(dataset, image_size):
    root = join(data_dir, dataset)
    if not os.path.exists(root):
        os.makedirs(root)
    if dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=root,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        nc = 3
    elif dataset == 'lsun':
        dataset = dset.LSUN(root=root, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        nc = 3
    elif dataset == 'cifar10':
        dataset = dset.CIFAR10(root=root, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        nc = 3

    elif dataset == 'mnist':
        dataset = dset.MNIST(root=root, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(image_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,)),
                             ]))
        nc = 1

    elif dataset == 'fake':
        dataset = dset.FakeData(image_size=(3, image_size, image_size),
                                transform=transforms.ToTensor())
        nc = 3
    else:
        raise ValueError(f'Wrong dataset, got {dataset}')
    return dataset, nc


@exp.main
def single(batch_size, image_size, test_samples, dataset, compute_fid, compute_is,
           n_jobs, device, nz, ngf, ndf, sampling, lr_decay, architecture, variance_reduction,
           lr, G_lr_factor, beta1, beta2, n_iter, loss_type, clip_value, print_every,
           importance_sampling, extrapolation, D_importance, G_importance, record_every,
           grad_penalty, checkpoint, optimizer, D_batch_norm, _run, _seed):
    if compute_fid and not compute_is:
        raise ValueError
    assert architecture in ['dcgan', 'resnet']
    if importance_sampling:
        assert sampling in ['one', 'bernouilli', 'half_alternated']
    else:
        assert sampling in ['all', 'alternated', 'half_alternated', 'one', 'bernouilli']

    # System
    torch.manual_seed(_seed)
    random.seed(_seed)
    np.random.seed(_seed)
    # Invert for deterministic results, but twice as slow
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    device = torch.device(device)

    artifact_dir = join(_run.observers[0].dir, 'artifacts')
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    # Data
    dataset_, nc = make_dataset()

    if compute_fid:
        activation_file = join(stat_dir, f'{dataset}.npy')
        if not os.path.exists(activation_file):
            dataloader = torch.utils.data.DataLoader(dataset_, batch_size=len(dataset_), shuffle=False,
                                                     num_workers=n_jobs)
            true_images = next(iter(dataloader))[0]
            true_images = true_images.add(1).mul_(255 / 2).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
            ref_activations = compute_activations(true_images)
            np.save(activation_file, ref_activations)
        else:
            ref_activations = np.load(activation_file)
    else:
        ref_activations = None

    dataloader = torch.utils.data.DataLoader(dataset_, batch_size=batch_size, shuffle=True, num_workers=n_jobs)
    dataloader_iter = infinite_iter(dataloader)

    fixed_noise = torch.randn(test_samples, nz, 1, 1, device=device)

    true_data = next(iter(dataloader))[0]
    vutils.save_image(true_data, join(artifact_dir, 'real_samples.png'), normalize=True)

    # Models
    if image_size == 64:
        if architecture == 'dcgan':
            net_G = GeneratorDCGAN64(in_features=nz, out_channels=nc, n_filters=ngf,
                                     batch_norm=True).to(device)
            net_D = DiscriminatorDCGAN64(in_channels=nc, n_filters=ndf, batch_norm=D_batch_norm).to(device)
        else:
            raise ValueError
    elif image_size == 32:
        if architecture == 'dcgan':
            net_G = GeneratorDCGAN32(in_features=nz, out_channels=nc, n_filters=ngf,
                                     batch_norm=True).to(device)
            net_D = DiscriminatorDCGAN32(in_channels=nc, n_filters=ndf, batch_norm=D_batch_norm).to(device)
        elif architecture == 'resnet':
            net_G = GeneratorResNet32(n_in=nz, n_out=nc, num_filters=ngf, batchnorm=True).to(device)
            net_D = DiscriminatorResNet32(n_in=nc, num_filters=ndf, batchnorm=D_batch_norm).to(device)
        else:
            raise ValueError(f'Wrong batch size, got {batch_size}')

    net_D.train()
    if loss_type == 'wgan':
        clip_weights(net_D, clip_value=clip_value)

    net_G.train()
    avg_net_G = copy.deepcopy(net_G)

    # Optimizers
    param_dict = {}
    if optimizer == 'adam':
        param_dict['betas'] = (beta1, beta2)

    optimizers = {'adam': Adam, 'sgd': SGD, 'rmsprop': RMSprop}
    optimizer_class = optimizers[optimizer]
    D_param_dict = copy.deepcopy(param_dict)
    G_param_dict = copy.deepcopy(param_dict)

    D_param_dict['lr'] = lr
    G_param_dict['lr'] = lr * G_lr_factor
    optimizer_D = optimizer_class(net_D.parameters(), **D_param_dict)
    optimizer_G = optimizer_class(net_G.parameters(), **G_param_dict)

    def lr_schedule(iter):
        if lr_decay:
            return 1 - iter / n_iter
        else:
            return 1.

    scheduler_D = LambdaLR(optimizer_D, lr_schedule)
    scheduler_G = LambdaLR(optimizer_G, lr_schedule)

    if extrapolation:
        if variance_reduction:
            optimizer_D = ExtraOptimizerVR(optimizer_D)
            optimizer_G = ExtraOptimizerVR(optimizer_G)
        else:
            optimizer_D = ExtraOptimizer(optimizer_D)
            optimizer_G = ExtraOptimizer(optimizer_G)

    if checkpoint is not None:
        load_checkpoint(net_G, net_D, avg_net_G, optimizer_G, optimizer_D, scheduler_D, scheduler_G, checkpoint)

    if sampling in ['bernouilli', 'one', 'half_alternated']:
        sampling_prob = np.array([D_importance, G_importance], dtype=float)
        sampling_prob /= np.sum(sampling_prob)
    else:
        # For print
        sampling_prob = np.array([.5, .5])
    upd_G, upd_D = False, True

    # Records
    D_iter, G_iter = 0, 0
    loss_D, loss_G, penalty = 0., 0., 0.
    period_loss_D, period_loss_G, period_penalty = 0., 0., 0.
    period_D_size, period_G_size = 0, 0
    iteration = 0
    elapsed_time = 0
    cur_elapsed_time = 0
    period_gnorm_D, period_gnorm_G = 0., 0.
    gnorm_D, gnorm_G = 0., 0.
    sum_iteration_times = {'D': 0, 'G': 0, 'DG': 0}
    avg_iteration_times = {'D': 0, 'G': 0, 'DG': 0}
    count_iterations = {'D': 0, 'G': 0, 'DG': 0}
    next_record_iter = 0
    next_print_iter = 0

    while G_iter < n_iter:
        if G_iter == next_print_iter:
            next_print_iter += print_every
            avg_loss_D = period_loss_D / max(1, period_D_size)
            avg_gnorm_D = period_gnorm_D / max(1, period_D_size)
            avg_penalty = period_penalty / max(1, period_D_size)
            avg_loss_G = period_loss_G / max(1, period_G_size)
            avg_gnorm_G = period_gnorm_G / max(1, period_G_size)

            str = (f"[i {iteration:05d}][D/G {D_iter:05d}/{G_iter:05d}]"
                   f" loss_D: {avg_loss_D:.4f} Loss_G: {avg_loss_G:.4f}")
            if penalty > 0:
                str += f" penalty {avg_penalty:.4f}\n"
            str += ' ' * 26
            str += f' |gD|: {avg_gnorm_D:.4f} |gG|: {avg_gnorm_G:.4f}' \
                f' |sD|: {sampling_prob[0]:.2f} |sG|: {sampling_prob[1]:.2f}'
            str += f" iter time: {cur_elapsed_time:.2f} s"
            print(str)

            _run.log_scalar('loss_D', avg_loss_D, iteration)
            _run.log_scalar('grad_D', avg_gnorm_D, iteration)
            _run.log_scalar('penalty', penalty, iteration)
            _run.log_scalar('loss_G', avg_loss_G, iteration)
            _run.log_scalar('grad_G', avg_gnorm_G, iteration)
            _run.log_scalar('loss_G_vs_time', avg_loss_G, elapsed_time)
            _run.log_scalar('loss_G_vs_G_iter', avg_loss_G, G_iter)
            _run.log_scalar('G_iter', G_iter, iteration)
            _run.log_scalar('D_iter', D_iter, iteration)
            _run.log_scalar('elapsed_time', elapsed_time, iteration)

            period_loss_D, period_loss_G, period_penalty = 0., 0., 0.
            period_gnorm_D, period_gnorm_G = 0., 0.
            period_D_size, period_G_size = 0, 0
            cur_elapsed_time = 0

        if G_iter == next_record_iter:
            t0 = time.perf_counter()
            next_record_iter += record_every
            _run.info['avg_iteration_times'] = avg_iteration_times

            with torch.no_grad():
                fake_images = avg_net_G(fixed_noise[:64]).cpu()
            vutils.save_image(fake_images,
                              join(artifact_dir, f'fake_samples_epoch_{G_iter}.png'), normalize=True)
            str = ' ' * 26

            if compute_is:
                fake_images = generate(fixed_noise, avg_net_G)
                # Free memory
                for var in [avg_net_G, net_G, net_D, fixed_noise]:
                    var.cpu()
                res = compute_scores(fake_images, ref_activations=ref_activations,
                                     compute_fid=compute_fid, splits=10, verbose=False)
                if compute_fid:
                    is_, is_std, fid = res
                else:
                    is_, is_std = res
                _run.log_scalar('is', is_, iteration)
                _run.log_scalar('is_std', is_std, iteration)
                _run.log_scalar('is_std_vs_time', is_std, elapsed_time)
                _run.log_scalar('is_std_vs_G_iter', is_std, G_iter)
                _run.log_scalar('is_vs_time', is_, elapsed_time)
                _run.log_scalar('is_vs_G_iter', is_, G_iter)
                if compute_fid:
                    _run.log_scalar('fid', fid, iteration)
                    _run.log_scalar('fid_vs_time', fid, elapsed_time)
                    _run.log_scalar('fid_vs_G_iter', fid, G_iter)
                _run.log_scalar('eval_elapsed_time', elapsed_time, iteration)
                str += f' IS: {is_:.4f}'
                if compute_fid:
                    str += f' FID: {fid:.4f}'
                for var in [avg_net_G, net_G, net_D, fixed_noise]:
                    var.to(device)

            save_checkpoint(net_G, net_D, avg_net_G, optimizer_G, optimizer_D, scheduler_G, scheduler_D,
                            join(artifact_dir, f'checkpoint_{iteration}.pth'))
            eval_time = time.perf_counter() - t0
            str += f' eval time {eval_time:.2f} s'
            print(str)

        t0 = time.perf_counter()

        if importance_sampling and iteration > 1:
            sampling_prob[0] = math.sqrt(gnorm_D)
            sampling_prob[1] = math.sqrt(gnorm_G)
            sampling_prob /= np.sum(sampling_prob)
            if sampling_prob[0] < .1:
                sampling_prob = torch.tensor([.1, .9])
            elif sampling_prob[1] < .1:
                sampling_prob = torch.tensor([.9, .1])

        extrapolate = extrapolation and iteration % 2 == 0
        update = not extrapolate
        if sampling == 'alternated':
            if extrapolation:
                if update:
                    upd_D, upd_G = upd_G, upd_D
                else:
                    if (iteration / 2) % (D_importance + G_importance) < D_importance:
                        upd_D, upd_G = True, False
                    else:
                        upd_G, upd_G = False, True
                # else upd_D, upd_G = upd_D, upd_G
            else:
                if iteration % (D_importance + G_importance) < D_importance:
                    upd_D, upd_G = True, False
                else:
                    upd_G, upd_G = False, True
        elif sampling == 'bernouilli':
            while True:
                sampled = np.random.rand(2) < sampling_prob
                if sampled.sum() > 0:
                    break
            upd_D, upd_G = bool(sampled[0]), bool(sampled[1])
        elif sampling in ['one', 'half_alternated']:
            if sampling == 'half_alternated' and update:
                upd_D, upd_G = upd_G, upd_D
            else:
                upd_D = bool(np.random.rand(1) < sampling_prob[0])
                upd_G = not upd_D
        elif sampling == 'all':
            upd_D, upd_G = True, True
        else:
            raise ValueError(f'Wrong sampling, got {sampling}')

        for p in net_D.parameters():
            p.requires_grad = upd_D
        for p in net_G.parameters():
            p.requires_grad = upd_G

        if upd_D:
            data = next(dataloader_iter)
            true_data = data[0].to(device)
            true_logits = net_D(true_data)
            this_batch_size = true_data.shape[0]
        else:
            true_logits = None
            this_batch_size = batch_size

        noise = torch.randn(this_batch_size, nz, 1, 1, device=device)

        fake_data = net_G(noise)
        fake_logits = net_D(fake_data)

        loss_D, loss_G, _ = compute_gan_loss(true_logits, fake_logits, loss_type=loss_type,
                                             compute_D=upd_D, compute_G=upd_G)

        if upd_D:
            net_D.zero_grad()
            for p in net_G.parameters():
                p.requires_grad = False
            loss_D.backward(retain_graph=True)
            loss_D = loss_D.item()

            if loss_type == 'wgan-gp':
                penalty = grad_penalty * compute_grad_penalty(net_D, true_data, fake_data)
                penalty.backward()
                penalty = penalty.item()
                loss_D += penalty

            for p in net_G.parameters():
                p.requires_grad = upd_G

            if extrapolation:
                optimizer_D.step(extrapolate=extrapolate)
                if variance_reduction and not upd_G:
                    optimizer_G.step(extrapolate=extrapolate)
            else:
                optimizer_D.step()
            if loss_type == 'wgan':
                clip_weights(net_D, clip_value=clip_value)

            if update:
                D_iter += 1

                gnorm_D *= beta2
                gnorm_D += compute_grad_norm(net_D) * (1 - beta2)

                period_gnorm_D += math.sqrt(gnorm_D) * this_batch_size
                period_loss_D += loss_D * this_batch_size
                period_D_size += this_batch_size

                if loss_type == 'wgan-gp':
                    period_penalty += penalty * this_batch_size

        if upd_G:
            net_G.zero_grad()
            for p in net_D.parameters():
                p.requires_grad = False
            loss_G.backward()
            loss_G = loss_G.item()
            # No need to reflag the parameters of net_D

            if extrapolation:
                optimizer_G.step(extrapolate=extrapolate)
                if variance_reduction and not upd_D:
                    optimizer_D.step(extrapolate=extrapolate)
                    if loss_type == 'wgan':
                        clip_weights(net_D, clip_value=clip_value)
            else:
                optimizer_G.step()

            if update:
                gnorm_G *= beta2
                gnorm_G += compute_grad_norm(net_G) * (1 - beta2)

                period_loss_G += loss_G * this_batch_size
                period_gnorm_G += math.sqrt(gnorm_G) * this_batch_size
                period_G_size += this_batch_size

                G_iter += 1
                update_avg(net_G, avg_net_G, 1. / G_iter)
                scheduler_G.step(G_iter)
                scheduler_D.step(G_iter)

        if extrapolation and update:
            # Reset extrapolated optimizers
            optimizer_D.deextrapolate()
            optimizer_G.deextrapolate()

        iteration += 1
        iteration_time = time.perf_counter() - t0
        elapsed_time += iteration_time
        cur_elapsed_time += iteration_time

        # Benchmark
        if upd_D and upd_G:
            key = 'DG'
        elif upd_D:
            key = 'D'
        else:
            key = 'G'
        sum_iteration_times[key] += iteration_time
        count_iterations[key] += 1
        avg_iteration_times[key] = sum_iteration_times[key] / count_iterations[key]


if __name__ == '__main__':
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    exp.observers = [FileStorageObserver.create(exp_dir)]
    exp.run_commandline()
