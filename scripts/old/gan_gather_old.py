import json
import os
from os.path import expanduser, join

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

data_dir = expanduser('~/data/games_rl')
stat_dir = expanduser('~/data/games_rl/stats')
output_dir = expanduser('~/output/games_rl/gan_figure')

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

plots = ['is_vs_time']
fig, axes = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
colors = sns.color_palette("Blues", n_colors=3)
colors = {5e-5: colors[2], 8e-5: colors[1], 1e-4: colors[0]}
for exp_id in range(8):
    try:
        # exp_id = exps[exp]
        exp_dir = expanduser(f'~/output/games_rl/gan/{exp_id}')
        with open(join(exp_dir, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
        with open(join(exp_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        with open(join(exp_dir, 'run.json'), 'r') as f:
            run = json.load(f)
    except:
        continue
    if config['variance_reduction']:
        continue
    lr = config['lr'] * config['G_lr_factor']


if config['sampling'] == 'all':
    label = f'Baseline extragrad {lr:.0e}'
    color = 'red'
else:
    label = f'Alt. subsamped {lr:.0e} {exp_id}'
    color = colors[lr]

for ax, metric in zip(axes, ['is_vs_time', 'is_vs_G_iter']):
    scores = metrics[f'{metric}_values']
    steps = metrics[metric]['steps']
    steps = [0] + steps
    scores = [1.18] + scores
    ax.plot(steps, scores, label=label, color=color, linewidth=2)
    if metric == 'is_vs_time':
        ax.set_xlabel('Wall-clock time (s)')
    else:
        ax.set_xlabel('Generator updates')
    ax.set_ylabel('Inception score')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
    ax.set_ylim([4, 8.8])

axes[1].legend(fontsize=6)
sns.despine(fig, axes[0])
for ax in axes:
    ax.grid(axis='y')

plt.savefig(join(output_dir, 'all_sampling.pdf'))
