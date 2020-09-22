# Make sure that all packages are up to date
using DrWatson; @quickactivate
using Pkg; Pkg.update()

# Use parallelism
using Distributed
nthreads = 32 # Number of threads to use
addprocs(nthreads) # Add the threads as workers

# Load all needed packages on every worker
@everywhere using DrWatson
@everywhere quickactivate(@__DIR__)
@everywhere include(srcdir("gaussian", "gaussian_target.jl"))

# Create a list of parameters
exp_ps = Dict(
    :n_iters => 2000, # Number of iterations to run
    :n_runs => 10, # Number of repeated runs
    :dim => vcat(1:9, 10:10:99, 100:100:500), # Dimension of the target
    :n_particles => [nothing, 10, 20, 50, 100], # Number of particles used, nothing will give dim + 1
    :full_cov => [true, false], # If the covariance is identity or a full covariance with varying eigenvalues
    :gpf => true, # Run GaussParticle Flow
    :advi => true, # Run Black Box VI
    :steinvi => true, # Run Stein VI
    :cond1 => false, # Use preconditionning on b
    :cond2 => false, # Use preconditionning on A
    :seed => 42, # Seed for experiments
    :cb_val => nothing, # Callback values
    :opt => ADAGrad(0.1), # Common optimizer
)
ps = dict_list(exp_ps)
@info "Will now run $(dict_list_count(exp_ps)) simulations"
# run for each dict the simulation
pmap(run_gaussian_target, ps)
