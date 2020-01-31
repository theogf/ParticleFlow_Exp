from typing import Tuple, List

import numpy as np

from gaussian_particle_flow import particles_step, random_gaussian, GaussianDensity

SMALL_LEARNING_RATE_M = 0.000001


def line_search_on_learning_rate_c(gaussian: GaussianDensity, particles: np.ndarray) -> Tuple[float, float]:
    """
    Performs line search to find optimal learning rate, assuming zero mean for both gaussians and particles
    :return: optimal step size, and objective value at optimal step size
    """

    def objective(step_size: float):
        new_particles = particles_step(particles, SMALL_LEARNING_RATE_M, step_size, gaussian)
        result = gaussian.approx_free_energy_bound(new_particles)
        if np.isnan(result):
            print('Was NAN!')
        return result

    # learning rates around the optimum
    step_sizes = [-np.inf, 0.1, np.inf]
    # objective values for the learning rates
    obj_values = [np.inf, objective(0.1), np.inf]

    def compute_next_learning_rate():
        """returns None after convergence"""

        assert len(step_sizes) == 3, step_sizes
        assert len(obj_values) == 3, obj_values
        assert obj_values[1] <= obj_values[0], obj_values
        assert obj_values[1] <= obj_values[2], obj_values
        assert step_sizes[0] <= step_sizes[1] <= step_sizes[2], step_sizes

        if np.isinf(step_sizes[0]):
            return step_sizes[1] / 10.0, 1

        if np.isinf(step_sizes[2]):
            return step_sizes[1] * 10.0, 2

        if (step_sizes[2] - step_sizes[0]) / step_sizes[2] < 0.1:
            # convergence condition
            return None, None

        if np.log(step_sizes[1]-step_sizes[0]) > np.log(step_sizes[2]-step_sizes[1]):
            return np.exp((np.log(step_sizes[0])+np.log(step_sizes[1]))/2), 1
        else:
            return np.exp((np.log(step_sizes[1])+np.log(step_sizes[2]))/2), 2

    while True:
        next_step_size, pos = compute_next_learning_rate()
        if next_step_size is None:
            break

        step_sizes.insert(pos, next_step_size)

        obj_values.insert(pos, objective(next_step_size))

        assert step_sizes[pos-1] <= step_sizes[pos] <= step_sizes[pos+1], step_sizes

        if obj_values[1] < obj_values[2]:
            del step_sizes[3]
            del obj_values[3]
        else:
            del step_sizes[0]
            del obj_values[0]

    return step_sizes[1], obj_values[1]


def optimize_with_optimal_rate_c(seed: int = 0, n_steps: int = 10, dim: int = 2)\
        -> Tuple[List[dict], GaussianDensity, List[np.ndarray]]:
    rand_state = np.random.RandomState(seed)
    gaussian = random_gaussian(dim, random_state=rand_state)
    print(gaussian.meta_data)
    num_particles = gaussian.dim() + 1

    data = []

    particles = rand_state.randn(num_particles, gaussian.dim())*rand_state.randn()
    particles -= particles.mean(axis=0)
    paths = [particles]

    for i in range(n_steps):
        lr, free_energy = line_search_on_learning_rate_c(gaussian, particles)
        new_particles = particles_step(particles, SMALL_LEARNING_RATE_M, lr, gaussian)
        new_data = {
            'seed': seed, 'lr': lr, 'num_particles': num_particles,
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
    return data, gaussian, paths


def find_optimal_learning_step_sizes_c(seed: int = 0, max_particle_exp: int = 10, dim: int = 2):
    """Finds optimal step size for different number of particles"""
    rand_state = np.random.RandomState(seed)
    gaussian = random_gaussian(dim, random_state=rand_state)
    print(gaussian.meta_data)
    num_particles = gaussian.dim() + 1
    data = []
    for i in range(1, max_particle_exp):
        print(f"computing optimal with {num_particles} particles")
        particles = rand_state.randn(num_particles, gaussian.dim())
        lr = line_search_on_learning_rate_c(gaussian, particles)
        new_data = {'seed': seed, 'lr': lr, 'num_particles': num_particles, 'num_particles_exp': i}
        new_data.update(gaussian.meta_data)
        data.append(new_data)
        num_particles = int((gaussian.dim() + 1) * 2**i)

    return data
