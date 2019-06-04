import os
import time
from os.path import expanduser

import numpy as np
from joblib import Memory
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.utils import check_random_state

from gamesrl.numpy.games import MatrixGame, make_positive_matrices
from gamesrl.numpy.schedulers import FullScheduler, AlternatedOneScheduler, BernouilliScheduler, RandomSubsetScheduler
from gamesrl.utils import softmax

exp = Experiment('matrix_games_clean')
exp_dir = expanduser(f'~/output/games_rl/{exp.path}')


@exp.config
def default():
    n_players = 5
    n_actions = 5
    n_matrices = 1
    gaussian_noise = 1
    stochastic_noise = 0

    n_iter = 100000
    eval_every = 5000
    step_size = .01
    averaging = 'uniform'
    schedule = 'constant'
    variance_reduction = False
    extrapolation = True
    sampling = 'alternated'
    conditioning = 0.01
    skewness = 0.9
    seed = 1000
    l1_penalty = 0.02

    training_seed = 100

    eval = {'step_size': 1e-2, 'tol': 1e-1, 'n_iter': 10000, 'extrapolation': True,
            'averaging': 'uniform', 'schedule': 'constant'}

@exp.capture(prefix='eval')
def compute_nash_gap(ref_policies, game: MatrixGame, init_policies=None, step_size=0.1, tol=1e-1,
                     extrapolation=False, n_iter=1000, averaging='uniform', schedule='1/sqrt(t)', warn=False):
    """Solve the Nash gap using mirror prox."""
    n_players, n_actions = ref_policies.shape
    step_size *= n_players
    ref_values = game.value(ref_policies)

    if init_policies is None:
        init_policies = ref_policies

    policies = np.maximum(1e-200, np.array(init_policies))
    policies /= np.sum(policies, axis=1, keepdims=True)
    log_policies = np.log(policies)
    adv_policies = np.array(ref_policies)

    if averaging in ['uniform', 'step_size']:
        avg_policies = np.array(policies)
        if averaging == 'step_size':
            total_step_size = 0
    else:
        avg_policies = policies

    grad = np.empty((n_players, n_actions))
    values = np.empty(n_players)

    if extrapolation:
        saved_log_policies = np.zeros_like(policies)
        saved_policies = np.zeros_like(policies)

    n_updates = 0
    this_step_size = step_size
    t = 0 # extrapolate clock

    while n_updates < n_iter:
        extrapolate = extrapolation and t % 2 == 0
        update = not extrapolate
        t += 1
        if extrapolate:
            saved_log_policies[:] = log_policies
            saved_policies[:] = policies
        elif extrapolation:  # Update step: go back to previously saved policies
            log_policies[:] = saved_log_policies
            policies[:] = saved_policies

        for i in range(n_players):
            adv_policies[i] = policies[i]
            grad[i] = game.gradient(adv_policies, index=i, random=False)  # Kill randomness to get exact bound
            values[i] = game.value(adv_policies, index=i)
            adv_policies[i] = ref_policies[i]

        log_policies -= grad * this_step_size
        policies = softmax(log_policies, axis=1)

        value_l = - float('inf')
        value_u = float('inf')

        if update:
            n_updates += 1
            if averaging == 'step_size':
                avg_policies *= total_step_size
                total_step_size += this_step_size
                avg_policies += policies * this_step_size
                avg_policies /= total_step_size
            elif averaging == 'uniform':
                avg_policies *= (1 - 1 / n_updates)
                avg_policies += policies / n_updates
            else:
                avg_policies = policies

            if schedule == '1/t':
                this_step_size = step_size / n_updates
            elif schedule == '1/sqrt(t)':
                this_step_size = step_size / np.sqrt(n_updates)
            elif schedule == 'constant':
                this_step_size = step_size
            else:
                raise ValueError('Wrong schedule argument')

            if n_updates % 100 == 0 or n_updates == n_iter - 1:
                for i in range(n_players):
                    adv_policies[i] = avg_policies[i]
                    grad[i] = game.gradient(adv_policies, index=i, random=False)  # Kill randomness to get exact bound
                    values[i] = game.value(adv_policies, index=i)
                    adv_policies[i] = ref_policies[i]
                new_value_l = np.sum(ref_values - values)
                err = np.sum(grad * avg_policies) - np.sum(np.min(grad, axis=1))
                new_value_u = new_value_l + err
                value_l = max(value_l, new_value_l)
                value_u = min(value_u, new_value_u)
                if value_u - value_l < (value_u + value_l) * tol:
                    break
    return value_l, value_u, avg_policies, n_updates


@exp.capture
def compute_nash(game: MatrixGame, _run, training_seed, extrapolation=True,
                 n_iter=100, step_size=1, eval_every=10,
                 averaging='uniform', sampling=1,
                 variance_reduction=False,
                 schedule='1/sqrt(t)'):
    random_state = check_random_state(training_seed)
    n_players, n_actions = game.n_players, game.n_actions
    step_size = float(step_size)

    log_policies = np.random.randn(n_players, n_actions)
    policies = softmax(log_policies, axis=1)

    if variance_reduction == 'auto':
        variance_reduction = game.l1_penalty == 0.
    if averaging in ['uniform', 'step_size']:
        n_updates = np.zeros(n_players)
        avg_policies = np.array(policies)
        if averaging == 'step_size':
            total_step_sizes = np.zeros(n_players)
    elif averaging == 'none' or averaging is False :
        avg_policies = policies
    else:
        raise ValueError(f'Wrong `averaging` argument, got {averaging}')

    if extrapolation:
        saved_policies = np.zeros_like(policies)
        saved_log_policies = np.zeros_like(log_policies)
        if variance_reduction:
            saved_grad = np.zeros_like(policies)
            full_grad = np.zeros_like(policies)

    if sampling == 'all':
        scheduler = FullScheduler(n_players, extrapolation)
        subsampling = 1
    elif sampling == 'alternated':
        scheduler = AlternatedOneScheduler(n_players, extrapolation, random_state)
        subsampling = 1. / n_players
    elif sampling == 'bernouilli':
        scheduler = BernouilliScheduler(n_players, extrapolation, random_state, batch_size=1)
        subsampling = 1. / n_players
    elif isinstance(sampling, int):
        player_batch_size = sampling
        scheduler = RandomSubsetScheduler(n_players, extrapolation, random_state, batch_size=player_batch_size)
        subsampling = player_batch_size / n_players
    else:
        raise ValueError(f'Wrong `sampling` argument, got {sampling}')

    step_sizes = np.full(n_players, step_size)

    elapsed_time = 0.
    n_computations = 0.
    next_eval_iter = 0
    # prev_policies = None
    while n_computations < n_iter:
        if eval_every and n_computations >= next_eval_iter:
            next_eval_iter += eval_every
            values = game.value(avg_policies)
            nash_gap_l, nash_gap_u, _, eval_iter = compute_nash_gap(avg_policies, game,
                                                                    init_policies=avg_policies)
            print(f'Iter {n_computations:.0f},'
                  f' nash gap {(nash_gap_l + nash_gap_u)/2:.1e} ± {(nash_gap_u - nash_gap_l)/2:.0e} (eval_iter {eval_iter})')
            for i, v in enumerate(values):
                _run.log_scalar(f'loss_{i}', v)
            _run.log_scalar('nash_gap_u_vs_computations', nash_gap_u, step=n_computations)
            _run.log_scalar('nash_gap_u_vs_time', nash_gap_u, step=elapsed_time)
            _run.log_scalar('nash_gap_l_vs_computations', nash_gap_l, step=n_computations)
            _run.log_scalar('nash_gap_l_vs_time', nash_gap_l, step=elapsed_time)

        t0 = time.perf_counter()

        mask, extrapolate = next(scheduler)
        update = not extrapolate
        mask = np.array(mask)

        n_computations += np.sum(mask) / n_players
        grad = game.gradient(policies, index=mask) / subsampling

        if extrapolate:
            saved_log_policies[:] = log_policies
            saved_policies[:] = policies
        elif extrapolation:
            log_policies[:] = saved_log_policies
            policies[:] = saved_policies

        if not variance_reduction:
            log_policies[mask] -= step_sizes[mask][:, None] * grad
            policies[mask] = softmax(log_policies[mask], axis=1)
        else:
            full_grad[:] = saved_grad
            full_grad[mask] *= subsampling - 1
            full_grad[mask] += grad
            saved_grad[mask] = grad
            log_policies -= step_sizes[:, None] * full_grad
            policies[:] = softmax(log_policies, axis=1)
        if update:
            if variance_reduction:
                mask = slice(None)

            n_updates[mask] += 1

            if averaging == 'step_size':
                avg_policies[mask] *= total_step_sizes[mask][:, None]
                total_step_sizes[mask] += step_sizes[mask]
                avg_policies[mask] += policies[mask] * step_sizes[mask][:, None]
                avg_policies[mask] /= total_step_sizes[mask][:, None]
            elif averaging == 'uniform':
                avg_policies[mask] *= (1 - 1 / (n_updates[mask][:, None] + 1))
                avg_policies[mask] += policies[mask] / (n_updates[mask][:, None] + 1)

            if schedule == '1/t':
                step_sizes = step_size / (n_updates + 1)
            elif schedule == '1/sqrt(t)':
                step_sizes = step_size / np.sqrt(n_updates + 1)

        elapsed_time += time.perf_counter() - t0

    return values, avg_policies


@exp.main
def single(n_players, n_actions, n_matrices, _seed, conditioning, skewness,  l1_penalty, gaussian_noise,
           stochastic_noise, _run):
    mem = Memory(location=expanduser('~/cache'))
    H = mem.cache(make_positive_matrices)(n_players, n_actions, n_matrices, conditioning, skewness,
                                          stochastic_noise, _seed)
    game = MatrixGame(H, l1_penalty=l1_penalty, gaussian_noise=gaussian_noise)

    values, policies = compute_nash(game)

    _run.info['policies'] = policies.tolist()
    _run.info['values'] = values.tolist()


if __name__ == '__main__':
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    exp.observers = [FileStorageObserver.create(exp_dir)]
    exp.run_commandline()
