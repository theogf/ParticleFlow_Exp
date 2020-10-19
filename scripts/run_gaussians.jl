# Make sure that all packages are up to date
using DrWatson;
@quickactivate
using Pkg; Pkg.update()
include(srcdir("gaussian", "gaussian_target.jl"))


# Set to true or false to use parallelism or not
do_parallel = true

# Use parallelism
if do_parallel
    using Distributed
    nthreads = 6 # Number of threads to use
    if nprocs() < nthreads
        addprocs(nthreads-nprocs()+1) # Add the threads as workers
    end
end

# Load all needed packages on every worker
if do_parallel
    @everywhere using DrWatson
    @everywhere quickactivate(@__DIR__)
    @everywhere include(srcdir("gaussian", "gaussian_target.jl"))
end

# Create a list of parameters
exp_ps = Dict(
    :n_iters => 3000, # Number of iterations to run
    :n_runs => 5, # Number of repeated runs
    :dim => vcat(2:9, 10:9:99, 100:100:500), # Dimension of the target
    :n_particles => [0, 10, 20, 50, 100], # Number of particles used, nothing will give dim + 1
    :full_cov => true,# false], # If the covariance is identity or a full covariance with varying eigenvalues
    :gpf => true, # Run GaussParticle Flow
    :advi => true, # Run Black Box VI
    :steinvi => true, # Run Stein VI
    :cond1 => false, # Use preconditionning on b
    :cond2 => false, # Use preconditionning on A
    :seed => 42, # Seed for experiments
    :cb_val => nothing, # Callback values
    # :opt => Flux.Optimise.Optimiser(ClipNorm(10.0), Descent(1.0)),#ADAGrad(0.1), # Common optimizer
    :opt => Flux.Optimise.Optimiser(ClipNorm(10), Descent(0.1)), # Common optimizer
)
ps = dict_list(exp_ps)
@info "Will now run $(dict_list_count(exp_ps)) simulations"

# run for each dict the simulation
if do_parallel
    pmap(run_gaussian_target, ps)
else
    map(run_gaussian_target, ps)
