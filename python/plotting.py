
import os

from gaussian_particle_flow import random_gaussian
from gradient_descent_only_c import move_particle_with_optimizer, Optimizer
from line_search import find_optimal_learning_rates_c, line_search_on_learning_rate, \
    optimize_with_optimal_rate_c, optimize_with_heuristic

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from second_order_learning_rate_heuristic import learning_rate_c_heuristic, \
    learning_rate_m_heuristic


def simple_gaussian_initial_rate_scatter():
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'

    data = []

    for dim in [2, 4, 8, 16, 32, 64]:
        for seed in range(8):
            x = find_optimal_learning_rates_c(seed, max_particle_exp=2, dim=dim)
            data.extend(x)
    df = pd.DataFrame(data)
    df.to_feather('data/simple_gaussian_initial_rate_scatter.ftr')

    df = pd.read_feather('data/simple_gaussian_initial_rate_scatter.ftr')
    print(data)
    df.lr = np.log10(df.lr)
    df.num_particles = np.log10(df.num_particles)
    df.inv_cond = np.log10(df.inv_cond)
    df.dim = np.log10(df.dim)
    g = sns.pairplot(df.drop(columns=['seed', 'num_particles', 'num_particles_exp']), kind='reg')
    plt.savefig('plots/simple_gaussian_initial_rate_scatter.png', bbox_inches='tight')
    plt.figure()


def simple_gaussian_increased_particles(regenerate_data=True):
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'

    if regenerate_data:
        data = []
        for dim in [2, 4, 8, 16, 32, 64]:
            for seed in range(8):
                x = find_optimal_learning_rates_c(seed, max_particle_exp=10, dim=dim)
                data.extend(x)
        df = pd.DataFrame(data)
        df.to_feather('data/simple_gaussian_increased_particles.ftr')

    df = pd.read_feather('data/simple_gaussian_increased_particles.ftr')
    df.lr = np.log10(df.lr)
    df.num_particles = np.log10(df.num_particles)
    df.inv_cond = np.log10(df.inv_cond)
    g = sns.FacetGrid(df, col='dim', row='seed', height=2)
    g = g.map(plt.plot, "num_particles_exp", 'lr', marker=".")
    plt.savefig('plots/simple_gaussian_increased_particles.png')


def plot_paths(gaussian, paths_list, filename):
    p = multivariate_normal(gaussian._m, gaussian._C).pdf

    x, y = np.mgrid[-2:2:.01, -2:2:.01]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    # compute path
    plt.figure(figsize=(20, 20))
    plt.contourf(x, y, -p(pos), alpha=0.4)
    for paths, color in zip(paths_list, 'bygr'):
        print(f"Printing in {color}")
        plt.plot(paths[0][:, 0], paths[0][:, 1], 'kx', alpha=0.7)
        plt.plot(paths[-1][:, 0], paths[-1][:, 1], f'{color}x', markersize=12, markeredgewidth=3)
        plt.ylim(-2, 2)
        plt.xlim(-2, 2)

        paths = np.array(paths)
        for i in range(paths.shape[1]):
            # plotting i-th particle
            plt.plot(paths[:, i, 0], paths[:, i, 1], color, linewidth=0.5)

    plt.savefig(filename, bbox_inches='tight')
    plt.figure()


def simple_gaussian_over_time(regenerate_data=True):
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'

    if regenerate_data:
        data = []
        for dim in [2, 4, 8, 16, 32, 64]:
            for seed in range(8):
                new_data, gaussian, paths = optimize_with_optimal_rate_c(seed=seed, n_steps=128, dim=dim)
                data.extend(new_data)
                if dim == 2:
                    plot_paths(gaussian, [paths], f'plots/simple_gaussian_over_time/{seed}.png')

        df = pd.DataFrame(data)
        df.to_feather('data/simple_gaussian_over_time.ftr')

    df = pd.read_feather('data/simple_gaussian_over_time.ftr')
    df.lr = np.log10(df.lr)
    df.num_particles = np.log10(df.num_particles)
    df.inv_cond = np.log10(df.inv_cond)
    df.free_energy = np.log10(df.free_energy)
    df.free_energy_d = np.log10(-df.free_energy_d)
    df.gradient_norm = np.log10(df.gradient_norm)
    df.n_step = np.log(df.n_step)

    g = sns.FacetGrid(df, col='dim', row='seed', height=1.5)
    g = g.map(plt.plot, "n_step", 'lr')
    plt.savefig('plots/simple_gaussian_over_time.png', bbox_inches='tight')
    plt.figure()

    g = sns.FacetGrid(df, col='dim', row='seed', height=1.5)
    g = g.map(plt.plot, "n_step", 'free_energy')
    plt.savefig('plots/simple_gaussian_over_time_free_energy.png', bbox_inches='tight')
    plt.figure()

    g = sns.FacetGrid(df, col='dim', row='seed', height=1.5)
    g = g.map(plt.plot, "n_step", 'gradient_norm')
    plt.savefig('plots/simple_gaussian_over_time_gradient.png', bbox_inches='tight')

    g = sns.FacetGrid(df, col='dim', row='seed', height=1.5)
    g = g.map(plt.plot, "n_step", 'free_energy_d')
    plt.savefig('plots/simple_gaussian_over_time_free_energy_d.png', bbox_inches='tight')
    plt.figure()


def simple_gaussian_over_time_momentum(initial_particles_mult=1.0, num_particles_mult=1.0):
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'

    base_path = f'plots/simple_gaussian_over_time_momentum_s{initial_particles_mult}_p{num_particles_mult}'
    os.makedirs(base_path, exist_ok=True)

    n_steps = 512
    for seed in range(8):
        rand_state = np.random.RandomState(seed)
        gaussian = random_gaussian(2, random_state=rand_state)
        num_particles = int((gaussian.dim() + 1)*num_particles_mult)
        particles = rand_state.randn(num_particles, gaussian.dim())*rand_state.randn()*initial_particles_mult
        particles -= particles.mean(axis=0)
        lr, _ = line_search_on_learning_rate(gaussian, particles)

        new_data, gaussian, paths = optimize_with_optimal_rate_c(seed=seed, n_steps=n_steps, dim=2)
        df = pd.DataFrame(new_data)
        df['optimiser'] = 'line search'
        data = [df]
        paths_list = [paths]
        for lr in [.1, .01]:
            for momentum in [.0, .8, .95]:
                identifier = f'lr={lr} m={momentum}'
                new_data, paths = move_particle_with_optimizer(
                    gaussian=gaussian,
                    optimizer=Optimizer(lr=lr*.5, momentum=momentum, identifier=identifier),
                    particles=particles,
                    n_steps=n_steps,
                )
                df = pd.DataFrame(new_data)
                df['optimiser'] = identifier
                data.append(df)
                # just first 64 steps of path
                paths_list.append(paths[:64])

        plot_paths(gaussian, paths_list, f'{base_path}/{seed}.png')
        data_df = pd.concat(data)
        data_df.free_energy = np.log10(data_df.free_energy)
        g = sns.lineplot(data=data_df, x='n_step', y='free_energy', hue='optimiser')
        g.set(ylim=(data_df.free_energy.min()-0.2, data_df[data_df.n_step == 0].free_energy.max() + .2))
        plt.savefig(f'{base_path}/f{seed}.png', bbox_inches='tight')
        plt.figure()
        data_df.free_energy_d = np.log10(-data_df.free_energy_d)
        g = sns.lineplot(data=data_df, x='n_step', y='free_energy_d', hue='optimiser')
        #g.set(ylim=(data_df.free_energy_d.min()-0.2, data_df[data_df.n_step == 0].free_energy_d.max() + .2))
        plt.savefig(f'{base_path}/df{seed}.png', bbox_inches='tight')
        plt.figure()


def simple_gaussian_over_time_heuristic():
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'

    base_path = f'plots/simple_gaussian_over_time_heuristic'
    os.makedirs(base_path, exist_ok=True)

    n_steps = 512
    for seed in range(8):
        rand_state = np.random.RandomState(seed)
        gaussian = random_gaussian(2, random_state=rand_state)
        num_particles = int((gaussian.dim() + 1))
        particles = rand_state.randn(num_particles, gaussian.dim())*rand_state.randn()
        particles -= particles.mean(axis=0)
        lr, _ = line_search_on_learning_rate(gaussian, particles)

        new_data, gaussian, paths = optimize_with_optimal_rate_c(seed=seed, n_steps=n_steps, dim=2)
        df = pd.DataFrame(new_data)
        df['optimiser'] = 'line search'
        data = [df]
        max_path_length = 64
        paths_list = [paths[:max_path_length]]

        new_data, paths = optimize_with_heuristic(seed=seed, n_steps=n_steps, dim=2)
        df = pd.DataFrame(new_data)
        df['optimiser'] = '2nd order'
        data.append(df)
        # just first 64 steps of path
        paths_list.append(paths[:max_path_length])

        plot_paths(gaussian, paths_list, f'{base_path}/{seed}.png')
        data_df = pd.concat(data)
        data_df.free_energy = np.log10(data_df.free_energy)
        g = sns.lineplot(data=data_df, x='n_step', y='free_energy', hue='optimiser')
        g.set(ylim=(data_df.free_energy.min()-0.2, data_df[data_df.n_step == 0].free_energy.max() + .2))
        plt.savefig(f'{base_path}/f{seed}.png', bbox_inches='tight')
        plt.figure()
        data_df.free_energy_d = np.log10(-data_df.free_energy_d)
        g = sns.lineplot(data=data_df, x='n_step', y='free_energy_d', hue='optimiser')
        #g.set(ylim=(data_df.free_energy_d.min()-0.2, data_df[data_df.n_step == 0].free_energy_d.max() + .2))
        plt.savefig(f'{base_path}/df{seed}.png', bbox_inches='tight')
        plt.figure()


def compare_learning_rate_heuristic_c():
    for seed in range(8):
        rand_state = np.random.RandomState(seed)
        gaussian = random_gaussian(2, random_state=rand_state)
        num_particles = gaussian.dim() + 1
        particles = rand_state.randn(num_particles, gaussian.dim())*rand_state.randn()
        particles -= particles.mean(axis=0)

        lr1, fe = line_search_on_learning_rate(gaussian, particles, tol=0.0001)

        lr2 = learning_rate_c_heuristic(gaussian, particles)
        print(lr1, lr2)


def compare_learning_rate_heuristic_m():
    for seed in range(8):
        rand_state = np.random.RandomState(seed)
        gaussian = random_gaussian(2, random_state=rand_state)
        num_particles = gaussian.dim() + 1
        mean = rand_state.gamma(gaussian.dim(), 2, size=(2,))
        particles = rand_state.multivariate_normal(mean, cov=np.eye(gaussian.dim()), size=(num_particles,))*rand_state.randn()
        particles += rand_state.randn(2)

        lr1, fe = line_search_on_learning_rate(gaussian, particles, tol=0.0001, learning_rate='m')

        lr2 = learning_rate_m_heuristic(gaussian, particles)
        print(lr1, lr2)


if __name__ == '__main__':
    #simple_gaussian_over_time_momentum(initial_particles_mult=.01, num_particles_mult=1.0)
    #compare_learning_rate_heuristic_c()
    #simple_gaussian_over_time_heuristic()
    compare_learning_rate_heuristic_m()


# try rprop rmsprop, adagrad, adam, on A?
# what about optimizing
# optimizing a regularizer (to avoid big values), or just use a max gradient value
