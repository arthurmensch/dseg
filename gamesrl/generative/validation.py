#!/usr/bin/env python3
"""Calculates metrics based on inception V3 model

Copyright 2018 Institute of Bioinformatics, JKU Linz
Modified by XXX

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from math import ceil

import numpy as np
import torch
from scipy import linalg
from torch.utils.data import DataLoader, TensorDataset
from .models import inception_v3, googlenet


def compute_statistics(model, dataset=None, noise=None, gen_model=None,
                       device='cpu', batch_size=50, n_jobs=4, verbose=0):
    if noise is None:
        n_samples = len(dataset)
        dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=n_jobs)
    else:
        n_samples = len(noise)
        dataset = TensorDataset(noise)
        dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
        gen_device = next(gen_model.parameters()).device

    n_batch = int(ceil(n_samples / batch_size))
    mu = torch.zeros((2048)).to(device)
    sigma = torch.zeros((2048, 2048)).to(device)
    preds = torch.empty((n_samples, 1000))

    model.to(device)
    model.eval()
    start = 0
    end = 0
    for i, data in enumerate(dataloader):
        if noise is not None:
            this_noise = data[0].to(gen_device)
            data = gen_model(this_noise).to(device)
        else:
            data = data[0].to(device)
        end = start + len(data)
        with torch.no_grad():
            these_logits, these_features = model(data)
        preds[start:end] = torch.softmax(these_logits, dim=1).cpu()
        mu += torch.sum(these_features, dim=0)
        sigma += torch.matmul(these_features.transpose(1, 0), these_features)
        start = end
        if verbose >= 10:
            print(f'[stat computation] iter {i}/{n_batch}')
    assert end == n_samples

    mean_pred = preds.mean(dim=0)
    mask = mean_pred != 0
    mean_pred = mean_pred[mask]
    preds = preds[:, mask]
    log_terms = (torch.log(preds) - torch.log(mean_pred)[None, :])
    log_terms[preds == 0] = 0
    kl = torch.sum(preds * log_terms) / n_samples
    is_ = torch.exp(kl).item()

    mu /= n_samples
    sigma -= torch.matmul(mu[:, None], mu[None, :]) * n_samples
    sigma /= n_samples - 1

    return mu.cpu(), sigma.cpu(), is_


class InceptionScorer:
    def __init__(self, device='cpu', verbose=False, compute_fid=True, n_jobs=4,
                 batch_size=50):
        self.device = device
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.compute_fid = compute_fid
        self.model = inception_v3(pretrained=True, transform_input=False)
        # self.model = googlenet(pretrained=True, transform_input=True)
        self.model.to(self.device)
        self.model.eval()

    def load_state_dict(self, state_dict):
        self.mu_ = state_dict['mu']
        self.sigma_ = state_dict['sigma']
        self.is_ = state_dict['is']

    def state_dict(self):
        if not hasattr(self, 'mu_'):
            raise ValueError('Fit estimator before')
        return {'mu': self.mu_, 'sigma': self.sigma_, 'is': self.is_}

    def fit(self, dataset):
        if not self.compute_fid:
            raise ValueError('Useless to fit is `compute_fid` is False')
        self.mu_, self.sigma_, self.is_ = compute_statistics(self.model, dataset=dataset,
                                                             device=self.device, batch_size=self.batch_size,
                                                             n_jobs=self.n_jobs, verbose=10)
        return self

    def compute(self, noise, gen_model):
        """Calculates the activations of the pool_3 layer for all images.
        """
        mu, sigma, is_ = compute_statistics(self.model, noise=noise, gen_model=gen_model,
                                            device=self.device, batch_size=self.batch_size, n_jobs=1,
                                            verbose=self.verbose)
        if self.verbose:
            print('Computing Frechet Inception Distance')
        if self.compute_fid:
            fid = calculate_frechet_distance(mu.numpy(), sigma.numpy(), self.mu_.numpy(), self.sigma_.numpy())
            return is_, fid
        return is_


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

