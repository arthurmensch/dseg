import os
import subprocess
from os.path import join
from tempfile import TemporaryDirectory

from joblib import dump, load

filedir = os.path.dirname(os.path.realpath(__file__))
tf_exec = join(filedir, '_tf_scores.py')


def compute_scores(images, compute_fid=False, fid_size=50000, ref_activations=None, splits=10,
                   verbose=True):
    with TemporaryDirectory() as temp_dir:
        images_filename = join(temp_dir, 'images.pkl')
        scores_filename = join(temp_dir, 'scores.pkl')
        dump(images, images_filename)
        if compute_fid:
            ref_activations_filename = join(temp_dir, 'ref_activations.pkl')
            dump(ref_activations, ref_activations_filename)
        else:
            ref_activations_filename = ''
        if not verbose:
            devnull = open(os.devnull, 'w')
            kwargs = dict(stderr=devnull, stdout=devnull)
        else:
            kwargs = dict()
        p = subprocess.run(["python", tf_exec, "scores", images_filename, scores_filename,
                            '--compute_fid', str(int(compute_fid)),
                            '--fid_size', str(fid_size),
                            '--ref_activations', ref_activations_filename,
                            '--splits', str(splits)],
                           **kwargs
                           )
        if not verbose:
            devnull.close()
        scores = load(scores_filename)
    return scores


def compute_activations(images, verbose=True):
    with TemporaryDirectory() as temp_dir:
        images_filename = join(temp_dir, 'images.pkl')
        dump(images, images_filename)
        activations_filename = join(temp_dir, 'activations.pkl')
        if not verbose:
            devnull = open(os.devnull, 'w')
            kwargs = dict(stderr=devnull, stdout=devnull)
        else:
            kwargs = dict()
        subprocess.run(["python", tf_exec, "activations", images_filename, activations_filename],
                       **kwargs
                       )
        if not verbose:
            devnull.close()
        activations = load(activations_filename)
    return activations
