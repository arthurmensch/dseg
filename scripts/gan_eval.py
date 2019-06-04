import glob
import json
import math
import os
from os.path import expanduser, join

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import transforms

from gamesrl.generative import GeneratorResNet32

data_dir = expanduser('~/data/games_rl')
stat_dir = expanduser('~/data/games_rl/stats')
output_dir = expanduser('~/output/games_rl/gan_figure')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def make_activation_file():
    from gamesrl.generative.scores import compute_activations
    activation_file = expanduser('~/data/games_rl/stats/cifar10.npy')
    dataset = torchvision.datasets.CIFAR10(root=expanduser('~/data/games_rl/cifar10'), download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize(32),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False,
                                             num_workers=4)
    true_images = next(iter(dataloader))[0]
    true_images = true_images.add(1).mul_(255 / 2).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    ref_activations = compute_activations(true_images)
    np.save(activation_file, ref_activations)


def convert_baseline():
    net_G = GeneratorResNet32(n_in=128, n_out=3, num_filters=128)
    checkpoint = expanduser('~/work/colab/joan/Variational-Inequality-GAN/results/ExtraAdam/best_model.state')
    with open(checkpoint, 'rb') as f:
        state_dict = torch.load(f)
    net_G.load_state_dict(state_dict['state_gen'], strict=False)
    for j, param in enumerate(net_G.parameters()):
        param.data = state_dict['gen_param_avg'][j]
    state_dict = {'avg_net_G': net_G.state_dict()}
    torch.save(state_dict, expanduser(join('~/output/games_rl/gan_figure', 'model_baseline.pkl')))


def score_checkpoint(checkpoint):
    from gamesrl.generative.scores import compute_scores
    device = 'cuda:0'
    test_samples = 50000

    batch_size = 64

    net_G = GeneratorResNet32(n_in=128, n_out=3, num_filters=128)

    activation_file = expanduser('~/data/games_rl/stats/cifar10.npy')
    ref_activations = np.load(activation_file)
    with open(checkpoint, 'rb') as f:
        state_dict = torch.load(f)
    net_G.load_state_dict(state_dict['avg_net_G'])

    net_G.train()
    net_G.to(device)

    noise = torch.randn(test_samples, 128, 1, 1, device=device)

    dataset = TensorDataset(noise)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    images = np.empty((test_samples, 3, 32, 32), dtype=np.uint8)

    n_batch = int(math.ceil(test_samples / batch_size))
    start, end = 0, 0
    for i, (this_noise,) in enumerate(dataloader):
        this_noise = this_noise.to(device)
        end = start + len(this_noise)
        with torch.no_grad():
            these_images = net_G(this_noise)
        images[start:end] = these_images.add(1).mul_(255 / 2).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
        start = end
        print(f'Generating batch {i}/{n_batch - 1}')
    return compute_scores(images, compute_fid=True, ref_activations=ref_activations, splits=10, verbose=False)


def gather_scores():
    res = []
    for score in glob.glob(join(output_dir, 'model_*.json')):
        if 'baseline' in score:
            model = 'gidel'
        elif 'all' in score:
            model = 'extragradient'
        elif 'alternated' in score:
            model = 'subsampled'
        else:
            continue
        with open(score, 'r') as f:
            this_res = json.load(f)
        this_res['model'] = model
        res.append(this_res)
    res = pd.DataFrame(res)
    res = res.groupby(by='model').agg(['mean', 'std'])
    res = res.round(2)
    print(res)


convert_baseline()
for checkpoint in glob.glob(join(output_dir, 'model_*.pkl')):
    is_, std, fid = score_checkpoint(checkpoint)
    with open(checkpoint.replace('.pkl', '_score.json'), 'w+') as f:
        json.dump(dict(is_=float(is_), std=float(std), fid=float(fid)), f)
gather_scores()
