from abc import ABC, abstractmethod

import numpy as np


class UnnormalizedDensity(ABC):
    def __call__(self, x):
        pass

    @abstractmethod
    def f(self, x):
        """Gradient of log(-p_tilde)"""
        pass

    @abstractmethod
    def dim(self) -> int:
        pass

    @abstractmethod
    def phi(self, x):
        pass

    def approx_free_energy_bound(self, x):
        assert x.shape[1] == self.dim(), x.shape
        C = np.cov(x.T, bias=True) + np.eye(self.dim())*1e-6
        log_determinant = np.log(abs(np.linalg.det(C)))
        return -.5 * log_determinant + np.apply_along_axis(self.phi, 1, x).mean(axis=0)


class GaussianDensity(UnnormalizedDensity):
    def __init__(self, m: np.ndarray, C: np.ndarray, meta_data=None):
        if len(m.shape) != 1:
            raise ValueError(f"m should be one dimensional, instead shape was {m.shape}")
        dim, = m.shape
        if C.shape != (dim, dim):
            raise ValueError(f"C should be one {dim}x{dim}, instead shape was {C.shape}")
        if meta_data is None:
            meta_data = {}
        self._dim = dim
        self._m = m
        self._C = C
        self._C_inv = np.linalg.inv(C)
        self.meta_data = meta_data

    def f(self, x):
        # -.5 gradient of .5*(x-m)'sigma-2(x-m)
        return -.5 * self._C_inv.dot(x-self._m)

    def phi(self, x):
        return .5 * (x - self._m).T.dot(self._C_inv).dot(x - self._m)

    def dim(self) -> int:
        return self._dim


def particles_gradients(x: np.ndarray, p_tilde: UnnormalizedDensity):
    M, D = x.shape

    mean = x.mean(axis=0)

    b = np.apply_along_axis(p_tilde.f, 1, x).mean(axis=0)
    A = np.zeros((D, D))
    for x_k in x:
        A += np.outer(p_tilde.f(x_k), x_k - mean)
    A /= M
    A += 1/2 * np.eye(D)

    return A, b


def particles_step(x: np.ndarray, learning_rate_m: float, learning_rate_c: float, p_tilde: UnnormalizedDensity):
    """
    :param x: MxD matrix, where M is the number of particles, and D the dimensionality of p
    :param learning_rate_m: learning rate of the first term
    :param learning_rate_c: learning rate of the second term
    :param p_tilde: wraps logic to compute free energy and related quantities
    :return:
    """

    mean = x.mean(axis=0)
    A, b = particles_gradients(x, p_tilde)

    new_x = []

    C = np.cov(x.T, bias=True)
    for k, x_k in enumerate(x):
        new_x_k = x_k + learning_rate_m * b + learning_rate_c * A@(x_k - mean)
        new_x.append(new_x_k)

    new_x = np.vstack(new_x)
    assert new_x.shape == x.shape, (x.shape, new_x.shape)
    return new_x


def random_gaussian(dim, random_mean=False, random_state: np.random.RandomState=None):
    """
    Generate random Gaussian

    Uses zero mean if random_mean is set to False (default)
    """
    if random_state is None:
        random_state = random_state.RandomState(0)
    random_matrix = random_state.uniform(size=(dim, dim))
    random_pos_definite_matrix = random_matrix.dot(random_matrix.T)
    eig, _ = np.linalg.eig(random_pos_definite_matrix)
    assert (eig > 0).all(), eig
    inv_cond = eig.min() / eig.max()
    if eig.max() == 0:
        inv_cond = 0
    if inv_cond < 0.0000001:
        print(f"random_gaussian: generated ill conditioned matrix of {dim} dims {eig.min()} {eig.max()}! Retrying...")
        return random_gaussian(dim, random_mean, random_state)
    if random_mean:
        mean = random_state.gamma(dim, 10)
    else:
        mean = np.zeros(dim)
    return GaussianDensity(mean, random_pos_definite_matrix, meta_data={'dim': dim, 'inv_cond': inv_cond})
