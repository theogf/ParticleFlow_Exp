import numpy as np

from gaussian_particle_flow import GaussianDensity, particles_gradients
from line_search import line_search_on_learning_rate_c, SMALL_LEARNING_RATE_M


class Optimizer:
    def __init__(self, lr: float, identifier: str, momentum=0.0):
        self.lr = lr
        self.identifier = identifier
        self.velocity = None
        self.momentum = momentum

    def reset(self):
        self.velocity = None

    def new_direction(self, gradient):
        gradient *= self.lr
        if self.velocity is None:
            self.velocity = gradient
        else:
            self.velocity = self.velocity*self.momentum + gradient
        return self.velocity


def move_particle_with_optimizer(gaussian: GaussianDensity, optimizer: Optimizer, particles: np.ndarray, n_steps: int):
    """
    Assumes Gaussian with zero mean
    """

    paths = [particles]

    # start with optimal learning rate
    lr, free_energy = line_search_on_learning_rate_c(gaussian, particles)

    optimizer.reset()
    data = []
    for i in range(n_steps):
        A, b = particles_gradients(particles, p_tilde=gaussian)
        A = optimizer.new_direction(A)

        new_particles = []

        for k, x_k in enumerate(particles):
            new_x_k = x_k + SMALL_LEARNING_RATE_M * b + A.dot(x_k - particles.mean(axis=0))
            new_particles.append(new_x_k)

        new_particles = np.vstack(new_particles)
        assert new_particles.shape == particles.shape, (particles.shape, new_particles.shape)

        new_data = {
            'lr': lr, 'num_particles': particles.shape[0],
            'n_step': i, 'free_energy': gaussian.approx_free_energy_bound(particles),
            'gradient_norm': np.linalg.norm((new_particles-particles).flatten())
        }
        if i != 0:
            data[-1]['free_energy_d'] = new_data['free_energy'] - data[-1]['free_energy']
        new_data.update(gaussian.meta_data)
        data.append(new_data)
        particles = new_particles
        paths.append(particles)

    data[-1]['free_energy_d'] = data[-2]['free_energy_d']
    return data, paths
