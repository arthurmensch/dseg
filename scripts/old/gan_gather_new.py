import json
import os
from os.path import expanduser, join

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

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

for exp_id in range(38, 76):
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
    sampling = config['sampling']
    lr = config['lr'] * config['G_lr_factor']
    label = f'sampl {sampling} {lr:.0e} {exp_id}'
    scores = metrics[metric]['values']
    steps = metrics[metric]['steps']


fig, axes = plt.subplots(1, 2, figsize=(12, 3), constrained_layout=True)

for ax, metric in zip(axes, ['is_vs_time', 'fid_vs_time']):
    scores = metrics[metric]['values']
    steps = metrics[metric]['steps']
    ax.plot(steps, scores, label=label, linewidth=2)
    if 'time' in metric:
        ax.set_xlabel('Wall-clock time (s)')
    else:
        ax.set_xlabel('Generator updates')
    ax.set_ylabel('Inception score')
    ax.tick_params(axis='x', labelsize=7)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
    if 'is' in metric:
        ax.set_ylim([4, 8.8])
    else:
        ax.set_ylim([15, 50])
axes[1].legend(fontsize=6)
sns.despine(fig, axes[0])
for ax in axes:
    ax.grid(axis='y')

plt.savefig(join(output_dir, 'metrics_new.pdf'))
