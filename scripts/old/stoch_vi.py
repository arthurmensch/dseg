import json
import os
import time
from os.path import join, expanduser
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed

from sklearn.utils import check_random_state


def logsumexp(a, axis=None):
    m = np.max(a, axis=axis)
    a = a - np.expand_dims(m, axis=axis)
    sumexp = np.sum(np.exp(a), axis=axis)
    return np.log(sumexp) + m

def mylogsumexp(a):
    z = np.zeros((len(a), 1))
    v = np.concatenate((z, a), 1)

    m = np.max(v, axis=1)
    v = v - np.expand_dims(m, axis=1)
    sumexp = np.sum(np.exp(v), axis=1)
    return np.sum(np.log(sumexp) + m)


def softmax(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    a = a - m
    r = np.exp(a)
    r /= np.sum(r, axis=axis, keepdims=True)
    return r


def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    return s


def logisticFunction(w, X, y, order):

    # w: d x 1
    # X: nData x d
    # y: nData x 1

    Xw = np.matmul(X,w)
    yXw = np.multiply(y, Xw) # Line by line multiplication

    if order == 0:
        out = mylogsumexp(-yXw)

    if order == 1:
        ratio = (y / (1+np.exp(yXw)))
        out = -np.matmul(X.T, ratio)
    return out


class MatrixGame:
    def __init__(self, matrix: np.ndarray, penalty=0.):
        self.out_features, self.in_features = matrix.shape[:2]  # Out_features: nPlayers, in features: dim of one player
        self.matrix = matrix
        self.jacobian = np.array(matrix)
        index = (range(self.out_features), slice(None),
                 range(self.out_features), slice(None))
        self.jacobian[index] += self.jacobian[index].transpose(0, 2, 1)

        self.penalty = penalty

    def value(self, policies, index: Union[int, slice] = slice(None)):
        if isinstance(index, int):
            index = [index]
        smooth = np.sum(np.einsum('ijkl,kl->ij', self.matrix[index], policies)
                        * policies[index], axis=1)
        sharp = policies[index] - 1 / self.in_features
        sharp = np.sum(np.abs(sharp), axis=1) * self.penalty
        return smooth + sharp

    def gradient(self, policies, index: Union[int, slice] = slice(None)):
        if isinstance(index, int):
            index = [index]
        smooth = np.einsum('ijkl,kl->ij', self.jacobian[index], policies)
        sharp = np.sign(policies[index] - 1 / self.in_features) * self.penalty
        return smooth + sharp


class LogisticGame:
    def __init__(self, dataLogistic: np.ndarray, labelLogistic: np.ndarray,
                 dataCoupling: np.ndarray, nPlayers, dim, penalty=1., coeffs=[1., 1., 1.]):

        # Out_features: nPlayers, in features: dim of one player
        self.out_features = nPlayers
        self.in_features = dim
        self.nPlayers = nPlayers
        self.dim = dim

        self.dataLogistic = dataLogistic
        self.labelLogistic = labelLogistic
        self.dataCoupling = dataCoupling
        self.penalty = penalty
        self.coeffs = coeffs

    def value(self, policies, index: Union[int, slice] = slice(None)):

        idx = [i for i in range(self.nPlayers)]
        index = idx[index]

        if isinstance(index, int):
            index = [index]

        fullVal = np.zeros(0)

        for idPlayer in index:

            if idPlayer == 0:
                others = policies[1:]
            elif idPlayer == len(policies)-1:
                others = policies[0:idPlayer]
            else:
                others = np.concatenate((policies[:idPlayer], policies[(idPlayer+1):]), 0)
            others = others.ravel()
            others = others.reshape(len(others), 1)

            player = policies[idPlayer]
            player = player.reshape(len(player), 1)

            valLogistic = logisticFunction(player, self.dataLogistic[idPlayer], self.labelLogistic[idPlayer], 0)
            valCoupling = np.matmul(player.T, self.dataCoupling[idPlayer])
            valCoupling = np.matmul(valCoupling, others)
            valCoupling = valCoupling.reshape(1)
            valNormOne = [self.penalty*np.sum(np.abs(player-1.0/self.dim))]

            val = np.multiply(self.coeffs[0], valLogistic) + np.multiply(self.coeffs[1], valCoupling) + np.multiply(self.coeffs[2], valNormOne)
            fullVal = np.concatenate((fullVal, val), axis=0)

        return fullVal

    def gradient(self, policies, index: Union[int, slice] = slice(None)):
        idx = [i for i in range(self.nPlayers)]
        if type(index) == type(np.zeros((0, 0))):
            index = np.where(index)
            index = index[0]
        else:
            index = idx[index]

        if isinstance(index, int):
            index = [index]

        fullGrad = np.zeros((0, self.dim))

        for idPlayer in index:

            if idPlayer == 0:
                others = policies[1:]
            elif idPlayer == len(policies) - 1:
                others = policies[0:idPlayer]
            else:
                others = np.concatenate((policies[:idPlayer], policies[(idPlayer + 1):]), 0)
            others = others.ravel()
            others = others.reshape(len(others), 1)

            player = policies[idPlayer]
            player = player.reshape(len(player), 1)

            gradLogistic = logisticFunction(player, self.dataLogistic[idPlayer], self.labelLogistic[idPlayer], 1)
            gradCoupling = np.matmul(self.dataCoupling[idPlayer], others)
            gradNormOne = self.penalty * np.sign(player-1.0/self.dim)

            grad = np.multiply(self.coeffs[0], gradLogistic) + np.multiply(self.coeffs[1], gradCoupling) + np.multiply(
                self.coeffs[2], gradNormOne)
            grad = grad.reshape(1, len(grad))
            fullGrad = np.concatenate((fullGrad, grad))

        return fullGrad


def make_positive_matrix(n_players, n_actions, cond=.1,
                         asym=.5, random_state=None):
    random_state = check_random_state(random_state)

    size = n_players * n_actions

    A = random_state.randn(size, size)
    A = .5 * (A + A.T)
    vs, _ = np.linalg.eigh(A)
    max_v = np.max(vs)
    A -= np.eye(size) * (np.min(vs) + max_v * cond)
    vs, _ = np.linalg.eigh(A)

    B = random_state.randn(size, size)
    B = .5 * (B - B.T)
    H = A * (1 - asym) + B * asym

    return H.reshape((n_players, n_actions, n_players, n_actions))


def solve_err_vi(ref_policies, game: MatrixGame, step_size=.1,
                 n_iter=500, averaging='step_size', schedule='1/sqrt(t)'):
    """Mirror descent for VI problem solving (less efficient than cvxopt)"""
    out_features, in_features = ref_policies.shape

    policies = np.array(ref_policies)
    log_policies = np.log(policies)

    avg_policies = np.zeros((out_features, in_features))
    total_step_size = 0.

    for t in range(n_iter):
        if schedule == '1/t':
            this_step_size = step_size / (t + 1)
        elif schedule == '1/sqrt(t)':
            this_step_size = step_size / np.sqrt(t + 1)
        elif schedule == 'constant':
            this_step_size = step_size
        else:
            raise ValueError('Wrong schedule argument')

        grad = game.gradient(ref_policies - 2 * policies)
        # print(np.sum(np.abs(grad - grad.mean(axis=1)[:, None])))
        log_policies += grad * this_step_size
        log_policies -= logsumexp(log_policies, axis=1)[:, None]
        policies = softmax(log_policies, axis=1)

        if averaging == 'step_size':
            avg_policies *= total_step_size
            total_step_size += this_step_size
            avg_policies += policies * this_step_size
            avg_policies /= total_step_size
        elif averaging == 'uniform':
            avg_policies *= (1 - 1 / (t + 1))
            avg_policies += policies / (t + 1)
        elif not averaging or averaging == 'none':
            avg_policies = policies
        else:
            raise ValueError('Wrong averaging argument')
    vi_gap = np.sum(game.gradient(avg_policies) * (ref_policies - avg_policies))

    return np.maximum(vi_gap, 1e-12), avg_policies


def solve_err_nash(ref_policies, game: MatrixGame, step_size=0.01,
                   n_iter=10000, averaging='step_size', schedule='constant'):
    """Mirror descent for VI problem solving (less efficient than cvxopt)"""
    out_features, in_features = ref_policies.shape

    policies = np.array(ref_policies)
    log_policies = np.log(policies)

    adv_policies = np.array(ref_policies)
    avg_policies = np.zeros((out_features, in_features))

    grad = np.empty((out_features, in_features))
    values = np.empty(out_features)

    total_step_size = 0.
    ref_values = game.value(ref_policies)

    for t in range(n_iter):
        if schedule == '1/t':
            this_step_size = step_size / (t + 1)
        elif schedule == '1/sqrt(t)':
            this_step_size = step_size / np.sqrt(t + 1)
        elif schedule == 'constant':
            this_step_size = step_size/np.sqrt(n_iter)
        else:
            raise ValueError('Wrong schedule argument')

        for i in range(out_features):
            adv_policies[i] = policies[i]
            grad[i] = game.gradient(adv_policies, index=i)
            adv_policies[i] = ref_policies[i]

        # print(np.sum(np.abs(grad - grad.mean(axis=1)[:, None])))

        extra_log_policies = log_policies - grad * this_step_size
        extra_policies = softmax(extra_log_policies, axis=1)

        for i in range(out_features):
            adv_policies[i] = extra_policies[i]
            grad[i] = game.gradient(adv_policies, index=i)
            adv_policies[i] = ref_policies[i]

        log_policies -= grad * this_step_size
        log_policies -= logsumexp(log_policies, axis=1)[:, None]
        policies = softmax(log_policies, axis=1)

        if averaging == 'step_size':
            avg_policies *= total_step_size
            total_step_size += this_step_size
            avg_policies += policies * this_step_size
            avg_policies /= total_step_size
        elif averaging == 'uniform':
            avg_policies *= (1 - 1 / (t + 1))
            avg_policies += policies / (t + 1)
        elif not averaging or averaging == 'none':
            avg_policies = policies
        else:
            raise ValueError('Wrong averaging argument')

    for i in range(out_features):
        adv_policies[i] = extra_policies[i]
        # values[i] = game.value(adv_policies, index=i)
        values[i] = game.value(avg_policies, index=i)
        adv_policies[i] = ref_policies[i]
        out = np.maximum(np.sum(ref_values - values), 1e-12), avg_policies
    return out


def solve_vi(game: MatrixGame, n_iter=100, step_size=1.,
             subsampling=1., history_file=None, random_state=None,
             print_every=10, averaging='step_size', schedule='1/sqrt(t)'):
    random_state = check_random_state(random_state)
    out_features, in_features = game.out_features, game.in_features

    log_policies = np.full((out_features, in_features), - np.log(in_features))
    policies = softmax(log_policies, axis=1)
    avg_policies = np.zeros_like(policies)

    timing = 0.
    gradient_computations = 0
    total_step_size = 0.

    values_r = []
    policies_r = []
    gradient_computations_r = []
    timings_r = []
    vi_gap_r = []
    nash_gap_r = []
    iterations_r = []

    avg_policies = policies

    for t in range(n_iter):
        if schedule == '1/t':
            this_step_size = step_size / (t + 1)
        elif schedule == '1/sqrt(t)':
            this_step_size = step_size / np.sqrt(t + 1)
        elif schedule == 'constant':
            this_step_size = step_size
        else:
            raise ValueError('Wrong schedule argument')

        if averaging == 'step_size':
            avg_policies *= total_step_size
            total_step_size += this_step_size
            avg_policies += policies * this_step_size
            avg_policies /= total_step_size
        elif averaging == 'uniform':
            avg_policies *= (1 - 1 / (t + 1))
            avg_policies += policies / (t + 1)
        elif not averaging or averaging == 'none':
            avg_policies = policies
        else:
            raise ValueError('Wrong averaging argument')

        if t % print_every == 0:
            # Value computation
            values = game.value(avg_policies)
            nash_gap, _ = solve_err_nash(policies, game, step_size)
            # vi_gap, _ = solve_err_vi(policies, game, step_size)
            vi_gap = 0

            print(f'Iter {t}, subsampling {subsampling}, vi gap {vi_gap},'
                  f' nash gap {nash_gap}')

            values_r.append(values.tolist())
            nash_gap_r.append(float(nash_gap))
            vi_gap_r.append(float(vi_gap))
            gradient_computations_r.append(int(gradient_computations))
            timings_r.append(float(timing))
            policies_r.append(policies.tolist())
            iterations_r.append(t)

        t0 = time.perf_counter()

        mask = random_state.binomial(1, subsampling,
                                     size=out_features).astype(np.bool)
        gradient_computations += np.sum(mask)
        mask = mask.astype(np.bool)
        if np.any(mask):
            grad = game.gradient(policies, index=mask)
            extra_log_policies = np.array(log_policies)
            extra_log_policies[mask] -= this_step_size * grad

            extra_policies = np.array(policies)
            extra_policies[mask] = softmax(extra_log_policies[mask], axis=1)
        else:
            extra_policies = policies

        mask = random_state.binomial(1, subsampling, size=out_features)
        gradient_computations += np.sum(mask)
        mask = mask.astype(np.bool)
        if np.any(mask):
            extra_grad = game.gradient(extra_policies, index=mask)
            log_policies[mask] -= this_step_size * extra_grad
            log_policies[mask] -= logsumexp(log_policies[mask], axis=1)[:, None]
            policies = softmax(log_policies, axis=1)

        timing += time.perf_counter() - t0

    history = {'values': values_r,
               'policies': policies_r,
               'vi_gap': vi_gap_r,
               'nash_gap': nash_gap_r,
               'gradient_computations': gradient_computations_r,
               'timings': timings_r,
               'iterations': iterations_r,
               'subsampling': subsampling,
               'n_players': out_features,
               }
    if history_file is not None:
        with open(history_file, 'w+') as f:
            json.dump(history, f)

    return values, avg_policies, history


def solve_vi_dual_averaging(game: MatrixGame, n_iter=100, step_size=1.,
             subsampling=1., history_file=None, random_state=None,
             print_every=10):
    random_state = check_random_state(random_state)
    out_features, in_features = game.out_features, game.in_features

    log_policies = np.full((out_features, in_features), - np.log(in_features))
    policies = softmax(log_policies, axis=1)
    avg_policies = np.zeros_like(policies)

    timing = 0.
    gradient_computations = 0
    total_step_size = 0.

    values_r = []
    policies_r = []
    gradient_computations_r = []
    timings_r = []
    vi_gap_r = []
    nash_gap_r = []
    iterations_r = []

    avg_policies = policies

    for t in range(n_iter):
        avg_policies *= (1 - 1 / (t + 1))
        avg_policies += policies / (t + 1)

        if t % print_every == 0:
            # Value computation
            values = game.value(avg_policies)
            nash_gap, _ = solve_err_nash(policies, game, step_size)
            # vi_gap, _ = solve_err_vi(policies, game, step_size)
            vi_gap = 0

            print(f'Iter {t}, subsampling {subsampling}, vi gap {vi_gap},'
                  f' nash gap {nash_gap}')

            values_r.append(values.tolist())
            nash_gap_r.append(float(nash_gap))
            vi_gap_r.append(float(vi_gap))
            gradient_computations_r.append(int(gradient_computations))
            timings_r.append(float(timing))
            policies_r.append(policies.tolist())
            iterations_r.append(t)

        t0 = time.perf_counter()

        mask = random_state.binomial(1, subsampling,
                                     size=out_features).astype(np.bool)
        gradient_computations += np.sum(mask)
        mask = mask.astype(np.bool)
        if np.any(mask):
            grad = game.gradient(policies, index=mask)
            extra_log_policies = np.array(log_policies)
            extra_log_policies[mask] -= this_step_size * grad

            extra_policies = np.array(policies)
            extra_policies[mask] = softmax(extra_log_policies[mask], axis=1)
        else:
            extra_policies = policies

        mask = random_state.binomial(1, subsampling, size=out_features)
        gradient_computations += np.sum(mask)
        mask = mask.astype(np.bool)
        if np.any(mask):
            extra_grad = game.gradient(extra_policies, index=mask)
            log_policies[mask] -= this_step_size * extra_grad
            log_policies[mask] -= logsumexp(log_policies[mask], axis=1)[:, None]
            policies = softmax(log_policies, axis=1)

        timing += time.perf_counter() - t0

    history = {'values': values_r,
               'policies': policies_r,
               'vi_gap': vi_gap_r,
               'nash_gap': nash_gap_r,
               'gradient_computations': gradient_computations_r,
               'timings': timings_r,
               'iterations': iterations_r,
               'subsampling': subsampling,
               'n_players': out_features,
               }
    if history_file is not None:
        with open(history_file, 'w+') as f:
            json.dump(history, f)

    return values, avg_policies, history


def plot_compare(output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    handles = []
    player_handles = []

    labels = []
    player_labels = []
    for index in range(4):
        with open(join(output_dir, f'history_{index}.json'), 'r') as f:
            res = json.load(f)
        timings = res['gradient_computations']
        values = res['values']
        p = res['subsampling']
        n_players = res['n_players']
        nash_gap = res['nash_gap']
        vi_gap = res['vi_gap']

        values = np.array(values)

        for player in range(n_players):
            cmap = sns.light_palette((23 * player, 90, 60), input="husl",
                                     n_colors=6, reverse=True)
            h, = axes[0].plot(timings, values[:, player], color=cmap[index],)
            if index == 0:
                player_handles.append(h)
                player_labels.append(f'Player {player}')
        cmap = sns.light_palette((0, 90, 60), input="husl",
                                 n_colors=6, reverse=True)
        # axes[1].plot(timings, vi_gap, color=cmap[index])
        h, = axes[1].plot(timings, nash_gap, color=cmap[index])
        handles.append(h)
        labels.append(f'p = {p:.2f}')

    # axes[1].set_yscale('log')
    axes[1].set_yscale('log')

    # axes[0].set_xscale('log')
    # axes[1].set_xscale('log')
    # axes[2].set_xscale('log')

    axes[0].set_xlabel('Computation')
    # axes[1].set_xlabel('Computation')
    axes[1].set_xlabel('Computation')
    # axes[1].set_ylabel('VI Gap')
    axes[1].set_ylabel('Nash Gap')

    fig.legend(handles, labels, ncol=1,
               bbox_to_anchor=[0.65, 1],
               loc='upper left', frameon=False)
    fig.legend(player_handles, player_labels, ncol=2,
               bbox_to_anchor=[0.08, 1],
               loc='upper left', frameon=False)

    sns.despine(fig)
    plt.savefig(join(output_dir, 'compare.pdf'))
    plt.show()


def randomLogisticGame(n_players, n_actions, nDataLogistic, penalty=1., coeffs=[1., 1., 1.]):

    dataLogistic = np.random.rand(n_players, nDataLogistic, n_actions)-0.5
    labelLogistic = np.round(np.random.rand(n_players, nDataLogistic, 1))*2-1
    dataCoupling = np.random.rand(n_players, n_actions, n_actions*(n_players-1))-0.5

    game = LogisticGame(dataLogistic, labelLogistic, dataCoupling, n_players, n_actions, penalty, coeffs)

    return game


def run_many(output_dir):
    n_players = 3
    n_actions = 2
    nDataLogistic = 10
    penalty = 1

    n_iter = 500
    print_every = 50
    step_size = .01

    # H = make_positive_matrix(n_players, n_actions, cond=0.1, asym=0.5, random_state=1)
    # game = MatrixGame(H, penalty=0.)

    coeffs = [1., 1., 0.1]
    game = randomLogisticGame(n_players, n_actions, nDataLogistic, penalty, coeffs)

    Parallel(n_jobs=1)(
        delayed(solve_vi)(game, n_iter=n_iter, step_size=step_size,
                          subsampling=subsampling, print_every=print_every,
                          history_file=join(output_dir, f'history_{i}.json', ),
                          schedule='constant', averaging='step_size',
                          random_state=1)
        for i, subsampling in enumerate([0.25, 0.5, 0.75, 1.]))

    # solve_vi(game, n_iter=n_iter, step_size=step_size,
    #     subsampling=0.25, print_every=print_every,
    #     history_file=join(output_dir, f'history_{0}.json', ),
    #     schedule='constant', averaging='step_size',
    #     random_state=0)

output_dir = expanduser('~/output/games_rl/subsampling_discontinuous')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

run_many(output_dir)
plot_compare(output_dir)
