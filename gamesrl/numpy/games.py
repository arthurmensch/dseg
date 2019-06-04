from typing import Union

import numpy as np
import scipy
from sklearn.utils import check_random_state


class MatrixGame:
    def __init__(self, matrix: np.ndarray, l1_penalty=0., gaussian_noise=0., random_state=None):
        if len(matrix.shape) == 4:
            matrix = matrix[None, :]
        self.n_matrices, self.n_players, self.n_actions = matrix.shape[:3]
        self.matrix = matrix
        self.average_matrix = matrix.mean(axis=0)
        self.jacobian = np.array(matrix)
        index = (slice(None), range(self.n_players), slice(None),
                 range(self.n_players), slice(None))
        self.jacobian[index] += self.jacobian[index].transpose(0, 1, 3, 2)
        self.average_jacobian = self.jacobian.mean(axis=0)

        self.l1_penalty = l1_penalty
        self.random_state = check_random_state(random_state)

        self.gaussian_noise = gaussian_noise

    def value(self, policies: np.ndarray, index: Union[int, slice] = slice(None)):
        if isinstance(index, int):
            index = [index]
            squeeze = True
        else:
            squeeze = False
        smooth = self.average_matrix[index].reshape((-1, self.n_players * self.n_actions)) @ \
                 policies.reshape((-1,))
        smooth = smooth.reshape(-1, self.n_actions)
        smooth = np.sum(smooth * policies[index], axis=1)

        sharp = policies[index] - 1 / self.n_actions
        sharp = np.sum(np.abs(sharp), axis=1) * self.l1_penalty

        res = smooth + sharp
        if squeeze:
            res = res[0]
        return res

    def gradient(self, policies: np.ndarray, index: Union[int, slice] = slice(None), random=True):
        if isinstance(index, int):
            index = [index]
            squeeze = True
        else:
            squeeze = False
        if random:
            jacobian = self.jacobian[self.random_state.randint(len(self.jacobian))]
        else:
            jacobian = self.average_jacobian

        smooth = jacobian[index].reshape((-1, self.n_players * self.n_actions)) @ policies.reshape((-1,))
        smooth = smooth.reshape(-1, self.n_actions)
        sharp = np.sign(policies[index] - 1 / self.n_actions) * self.l1_penalty
        if random:
            noise = self.gaussian_noise * self.random_state.randn(*smooth.shape)
        else:
            noise = 0
        res = smooth + sharp + noise
        if squeeze:
            res = res[0]
        return res

    @property
    def lipschitz(self):
        """Lipschitz constant of the smooth part."""
        return np.sum(np.max(np.abs(self.jacobian), axis=(1, 3)), axis=1)


def make_positive_matrix(n_players: int, n_actions: int,
                         conditioning: float = .1, skewness: float = .5, seed: int = None):
    """Generate a positive matrix."""
    random_state = check_random_state(seed)

    size = n_players * n_actions

    A = random_state.randn(size, size)
    A += A.T
    A /= 2
    vs = scipy.linalg.eigh(A, eigvals_only=True)
    A += np.eye(size) * (conditioning * (np.max(vs) - np.min(vs)) - np.min(vs))

    B = random_state.randn(size, size)
    B -= B.T
    B /= 2
    B *= skewness
    B += A * (1 - skewness)
    return B.reshape((n_players, n_actions, n_players, n_actions))


def make_positive_matrices(n_players: int, n_actions: int, n_matrices: int, conditioning: float = .1,
                           skewness: float = .5, noise: float = 10, seed: int = None):
    A = np.empty((n_matrices, n_players, n_actions, n_players, n_actions))
    random_state = check_random_state(seed)
    for i in range(n_matrices):
        this_seed = random_state.randint(0, np.iinfo(np.int32).max)
        A[i] = make_positive_matrix(n_players, n_actions, conditioning, skewness, this_seed)
    A *= noise
    A += make_positive_matrix(n_players, n_actions, conditioning, skewness,
                              random_state.randint(0, np.iinfo(np.int32).max))
    return A


def make_rps():
    matrix = np.zeros((2, 3, 2, 3))
    for i in range(2):
        matrix[i, :, (i + 1) % 2, :] = np.array(([[0, -1., 1], [1, 0, -1], [-1, 1, 0]]))
    print(matrix.reshape((6, 6)))
    return matrix


def make_mp():
    matrix = np.zeros((2, 2, 2, 2))
    for i in range(2):
        matrix[i, :, (i + 1) % 2, :] = np.array(([[1, -1], [-1, 1]]))
    return matrix
