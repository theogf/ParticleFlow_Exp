# Make sure that all packages are up to date
using DrWatson
@quickactivate
# using Pkg; Pkg.update()
include(srcdir("gaussian", "gaussian_target.jl"))

# Use parallelism
# using Distributed
# nthreads = 60 # Number of threads to use
# nthreads = min(nthreads, Sys.CPU_THREADS - 1)
# if nprocs() < nthreads
    # addprocs(nthreads - nprocs() + 1) # Add the threads as workers
# end

# Load all needed packages on every worker
# @everywhere using DrWatson
# @everywhere quickactivate(@__DIR__)
# @everywhere include(srcdir("gaussian", "gaussian_target.jl"))
# Create a list of parameters
exp_ps = Dict(
    :n_iters => 20000, # Number of iterations to run
    :n_runs => 10, # Number of repeated runs
    :n_dim => [20, 50],# 100, 500], # Dimension of the target
    :n_particles => [collect(2:11)..., @onlyif(:n_dim > 10, collect(20:50))..., @onlyif(:n_dim > 50, collect(60:10:100))..., @onlyif(:n_dim > 100, collect(200:100:500))...],
    :cond => [1, 5, 10, 50, 100],
    :gpf => true, # Run GaussParticle Flow
    :gf => false, # Run Gauss Flow
    :dsvi => false, # Run Doubly Stochastic VI
    :fcs => false, # Run Factorized Structure Covariance
    :iblr => false, # Run i Bayesian Rule
    :svgd => false, # Run linear SVGD
    :natmu => false, # Use preconditionning on b
    :seed => 42, # Seed for experiments
    :cb_val => nothing, # Callback values
    :eta => 0.001,
    :opt_det => :Descent,
    :opt_stoch => :Descent,
    :comp_hess => :hess,
    :partial => true,
    :overwrite => true,
)
ps = dict_list(exp_ps)
@info "Will now run $(dict_list_count(exp_ps)) simulations"
# run for each dict the simulation
# run_gaussian_target(ps[1])
map(run_gaussian_target, ps)
# pmap(run_gaussian_target, ps)
