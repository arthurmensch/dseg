import glob
import json
import os
from os.path import expanduser, join

import numpy as np
import torch

from models import ResNet32Generator


data_dir = expanduser('~/data/games_rl')
stat_dir = expanduser('~/data/games_rl/stats')
output_dir = expanduser('~/output/games_rl/gan_figure')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def score_checkpoint_gidel(input_path):
    CUDA = True
    BATCH_SIZE = 32
    N_CHANNEL = 3
    RESOLUTION = 32
    NUM_SAMPLES = 50000
    DEVICE = 'cpu'

    checkpoint = torch.load(input_path, map_location=DEVICE)

    N_LATENT = 128
    N_FILTERS_G = 128
    BATCH_NORM_G = True

    print "Init..."
    import tflib.fid as fid
    import tflib.inception_score as inception_score
    import tensorflow as tf
    stats_path = 'tflib/data/fid_stats_cifar10_train.npz'
    inception_path = fid.check_or_download_inception('tflib/model')
    f = np.load(stats_path)
    mu_real, sigma_real = f['mu'][:], f['sigma'][:]
    f.close()

    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    def get_fid_score():
        all_samples = []
        samples = torch.randn(NUM_SAMPLES, N_LATENT)
        for i in range(0, NUM_SAMPLES, BATCH_SIZE):
            samples_100 = samples[i:i+BATCH_SIZE]
            if CUDA:
                samples_100 = samples_100.cuda(0)
            all_samples.append(gen(samples_100).cpu().data.numpy())

        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
        all_samples = all_samples.reshape((-1, N_CHANNEL, RESOLUTION, RESOLUTION)).transpose(0, 2, 3, 1)

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            mu_gen, sigma_gen = fid.calculate_activation_statistics(all_samples, sess, batch_size=BATCH_SIZE)

        fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
        return fid_value

    def get_inception_score():
        all_samples = []
        samples = torch.randn(NUM_SAMPLES, 128)
        for i in range(0, NUM_SAMPLES, BATCH_SIZE):
            samples_100 = samples[i:i + BATCH_SIZE]
            if CUDA:
                samples_100 = samples_100.cuda(0)
            all_samples.append(gen(samples_100).cpu().data.numpy())

        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
        all_samples = all_samples.reshape((-1, N_CHANNEL, RESOLUTION, RESOLUTION)).transpose(0, 2, 3, 1)
        return inception_score.get_inception_score(list(all_samples))

    gen = ResNet32Generator(N_LATENT, N_CHANNEL, N_FILTERS_G, BATCH_NORM_G)

    print "Eval..."
    gen.load_state_dict(checkpoint['avg_net_G'], strict=False)
    if CUDA:
        gen.cuda(0)
    is_, std = get_inception_score()
    fid = get_fid_score()
    return is_, std, fid


# for checkpoint in glob.glob(join(output_dir, 'model_*.pkl')):
for checkpoint in [join(output_dir, 'model_baseline.pkl'), join(output_dir, 'model_alternated_18.pkl')]:
    is_, std, fid = score_checkpoint_gidel(checkpoint)
    with open(checkpoint.replace('.pkl', '_score_gidel.json'), 'w+') as f:
        json.dump(dict(is_=float(is_), std=float(std), fid=float(fid)), f)
