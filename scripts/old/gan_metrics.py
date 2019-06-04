import json
import os
from os.path import expanduser, join

import matplotlib.pyplot as plt

data_dir = expanduser('~/data/games_rl')
stat_dir = expanduser('~/data/games_rl/stats')
output_dir = expanduser('~/output/games_rl/gan/metrics')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

seed = 100
device = 'cuda:0'
test_samples = 50000

batch_size = 64
image_size = 32

architecture = 'resnet'
ngf = 128
ndf = 128
nz = 128
nc = 3

# exps = {'vanilla': 59, 'look_ahead': 57, 'slow_look_ahead': 52}

plots = ['grad_D', 'loss_D', 'grad_G', 'loss_G', 'penalty']
fig, axes = plt.subplots(2, 3, figsize=(10, 10))
axes = axes.ravel()
axes = {metric: ax for metric, ax in zip(plots, axes)}
for exp_id in range(18):
    try:
        # exp_id = exps[exp]
        exp_dir = expanduser(f'~/output/games_rl/gan/{exp_id}')
        with open(join(exp_dir, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
        with open(join(exp_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        with open(join(exp_dir, 'run.json'), 'r') as f:
            run = json.load(f)
        for metric, ax in axes.items():
            scores = metrics[metric]['values']
            steps = metrics[metric]['steps']
            ax.plot(steps[:100], scores[:100])
            ax.set_ylabel(metric)
            ax.set_xlabel('Iteration')
    except:
        continue
# axes[plots[0]].legend()
plt.savefig(join(output_dir, 'metrics.pdf'))
