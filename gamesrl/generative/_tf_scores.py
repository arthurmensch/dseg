'''
From https://github.com/tsc2017/Inception-Score
Code derived from https://github.com/openai/improved-gan/blob/master/inception_score/model.py and https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
Usage:
    Call get_inception_score(images, splits=10)
Args:
    images: A numpy array with values ranging from 0 to 255 and shape in the form [N, 3, HEIGHT, WIDTH] where N, HEIGHT and WIDTH can be arbitrary.
            dtype of the images is recommended to be np.uint8 to save CPU memory.
    splits: The number of splits of the images, default is 10.
Returns:
    Mean and standard deviation of the Inception Score across the splits.
'''
import argparse
import os
import time

import numpy as np
import tensorflow as tf
from gamesrl.utils import softmax
from joblib import load, dump
from sklearn.utils import gen_batches
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops

tfgan = tf.contrib.gan

BATCH_SIZE = 64

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.reset_default_graph()

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, )
config.gpu_options.allow_growth = True


def _run_inception(images):
    return tuple(tfgan.eval.run_inception(images, output_tensor=['logits:0', 'pool_3:0']))


input_images_ = tf.placeholder(tf.float32, [None, 3, None, None])
images_ = tf.transpose(input_images_, [0, 2, 3, 1])
size_ = 299
images_ = tf.image.resize_bilinear(images_, [size_, size_])
generated_images_list_ = array_ops.split(images_, num_or_size_splits=1)

with tf.device('/gpu:0'):
    # Run images through Inception.
    logits_, activations_ = functional_ops.map_fn(
        fn=_run_inception,
        elems=array_ops.stack(generated_images_list_),
        parallel_iterations=1,
        back_prop=False,
        dtype=(tf.float32, tf.float32),
        swap_memory=True,
        name='RunClassifier')

logits_ = array_ops.concat(array_ops.unstack(logits_), 0)
activations_ = array_ops.concat(array_ops.unstack(activations_), 0)

activations1_ = tf.placeholder(tf.float32, [None, None], name='activations1')
activations2_ = tf.placeholder(tf.float32, [None, None], name='activations2')
fcd = tfgan.eval.frechet_classifier_distance_from_activations(activations1_, activations2_)


def activations_to_distance(activations1, activations2):
    print('Compute distance')
    with tf.Session(config=config) as sess:
        return sess.run(fcd, feed_dict={activations1_: activations1, activations2_: activations2})


def get_inception_output(inps, compute_fid=False):
    n_samples = len(inps)
    batches = list(gen_batches(n_samples, BATCH_SIZE))
    n_batches = len(batches)
    # Thhis should be corrected to 1000 but everybody uses 1008
    preds = np.zeros([len(inps), 1000], dtype=np.float32)
    if compute_fid:
        activations = np.zeros([len(inps), 2048], dtype=np.float32)

    with tf.Session(config=config) as sess:
        for i, batch in enumerate(batches):
            inp = inps[batch] / 255. * 2 - 1
            if compute_fid:
                these_logits, these_activations = sess.run([logits_, activations_],
                                                           feed_dict={input_images_: inp})
                activations[batch] = these_activations
            else:
                these_logits = sess.run(logits_, feed_dict={input_images_: inp})
            preds[batch] = these_logits[:, :1000]
            if i % 100 == 0:
                print(f'inception network {i}/{n_batches}')
    preds = softmax(preds, axis=1)
    if compute_fid:
        return preds, activations
    else:
        return preds


def preds_to_score(preds, splits=10):
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def compute_activations(images, fid_size=50000):
    assert (type(images) == np.ndarray)
    assert (len(images.shape) == 4)
    assert (images.shape[1] == 3)
    assert (np.min(images[0]) >= 0 and np.max(images[0]) > 10), 'Image values should be in the range [0, 255]'
    print('Calculating activations')
    start_time = time.time()
    activations = get_inception_output(images, compute_fid=True)[1]
    activations = activations[:fid_size]
    print('Activation calculation time: %f s' % (time.time() - start_time))
    return activations


def compute_scores(images, compute_fid=False, fid_size=10000, ref_activations=None, splits=10):
    assert (type(images) == np.ndarray)
    assert (len(images.shape) == 4)
    assert (images.shape[1] == 3)
    assert (np.min(images[0]) >= 0 and np.max(images[0]) > 10), 'Image values should be in the range [0, 255]'
    print('Calculating Inception Score with %i images in %i splits' % (images.shape[0], splits))
    start_time = time.time()
    res = get_inception_output(images, compute_fid=compute_fid)
    if compute_fid:
        preds, activations = res
    else:
        preds = res
    mean, std = preds_to_score(preds, splits)
    res = [mean, std]  # Reference values: 11.34 for 49984 CIFAR-10 training set images,
    # or mean=11.31, std=0.08 if in 10 splits.
    if compute_fid:
        print('Calculating FID with %i images' % images.shape[0])
        fid = activations_to_distance(activations[:fid_size], ref_activations[:fid_size])
        res.append(fid)
    print('Scores calculation time: %f s' % (time.time() - start_time))
    return tuple(res)


parser = argparse.ArgumentParser()
parser.add_argument('command', choices=['scores', 'activations'])
parser.add_argument('images', type=str)
parser.add_argument('output', type=str)
parser.add_argument('--compute_fid', type=int, default=0)
parser.add_argument('--ref_activations', type=str)
parser.add_argument('--fid_size', type=int, default=10000)
parser.add_argument('--splits', type=int, default=10)
args = parser.parse_args()

if args.command == 'activations':
    images = load(args.images)
    activations = compute_activations(images)
    dump(activations, args.output)
elif args.command == 'scores':
    images = load(args.images)
    if args.compute_fid:
        ref_activations = load(args.ref_activations)
    else:
        ref_activations = None
    scores = compute_scores(images, args.compute_fid, args.fid_size, ref_activations, args.splits)
    dump(scores, args.output)