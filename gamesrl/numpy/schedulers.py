import numpy as np
from sklearn.utils import check_random_state


def FullScheduler(n_players, extrapolation):
    i = 0
    while True:
        extrapolate = i % 2 == 0 and extrapolation
        i += 1
        yield [True for _ in range(n_players)], extrapolate


def AlternatedOneScheduler(n_players, extrapolation, random_state=None, shuffle=True):
    random_state = check_random_state(random_state)
    if extrapolation:
        all_pairs = [(i, j) for i in range(n_players) for j in range(n_players) if i != j]
        if shuffle:
            random_state.shuffle(all_pairs)
        while True:
            for (i, j) in all_pairs:
                mask = [False for _ in range(n_players)]
                mask[i] = True
                yield mask, True
                mask[i] = False
                mask[j] = True
                yield mask, False
            if shuffle:
                random_state.shuffle(all_pairs)
    else:
        all_players = list(range(n_players))
        if shuffle:
            random_state.shuffle(all_players)
        while True:
            for i in all_players:
                mask = [False for _ in range(n_players)]
                mask[i] = True
                yield mask, False
            if shuffle:
                random_state.shuffle(all_players)


def BernouilliScheduler(n_players, extrapolation, random_state=None, batch_size=1):
    random_state = check_random_state(random_state)
    p = batch_size / n_players
    i = 0
    while True:
        extrapolate = i % 2 == 0 and extrapolation
        i += 1
        while True:
            mask = (random_state.uniform(size=n_players) < p)
            if mask.astype(int).sum() > 0:
                break
        yield mask.tolist(), extrapolate


def RandomSubsetScheduler(n_players, extrapolation, random_state=None, batch_size=1):
    random_state = check_random_state(random_state)
    i = 0
    while True:
        extrapolate = i % 2 == 0 and extrapolation
        i += 1
        indices = random_state.choice(n_players, size=batch_size, replace=False)
        mask = np.zeros(n_players, dtype=np.bool)
        mask[indices] = True
        yield mask.tolist(), extrapolate