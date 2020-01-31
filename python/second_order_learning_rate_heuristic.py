from typing import Tuple

import numpy as np

from gaussian_particle_flow import GaussianDensity


def learning_rate_c_heuristic(gaussian: GaussianDensity, particles: np.ndarray) -> float:
    """Second order heuristic for the learning rate for C"""
    B = gaussian._C_inv

    C = np.cov(particles.T, bias=True)
    A = np.eye(C.shape[0])*.5 - B@C*.5
    res = 2*np.trace(A.T@A) / np.trace(A@A + A.T@B@A@C)
    return res


def learning_rate_m_heuristic(
        gaussian: GaussianDensity, particles: np.ndarray, precondition_with_C: bool=False
) -> Tuple[float, np.array]:
    """Returns learning rate and gradient for mean component"""
    B = gaussian._C_inv
    C = np.cov(particles.T, bias=True)
    Ef = np.apply_along_axis(gaussian.f, 1, particles).mean(axis=0)
    if precondition_with_C:
        b = C*Ef
    else:
        b = Ef
    # had to add a 2* to make the results match the line search
    res = 2*b.T@Ef/(b.T@B@b)
    return res, b
