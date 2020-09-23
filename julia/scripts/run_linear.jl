# Make sure that all packages are up to date
using DrWatson;
@quickactivate
using Pkg; Pkg.update()

# Use parallelism
# using Distributed
# nthreads = 32 # Number of threads to use
# if nprocs() < nthreads
    # addprocs(nthreads-nprocs()+1) # Add the threads as workers
# end

# Load all needed packages on every worker
# @everywhere using DrWatson
# @everywhere quickactivate(@__DIR__)
# @everywhere include(srcdir("linear", "linear.jl"))

include(srcdir("linear", "linear.jl"))

# Create a list of parameters
exp_ps = Dict(
    :seed => 42,
    :dataset => "swarm_flocking",
    :n_iters => 200, # Number of iterations to run
    :n_particles => 10, # Number of particles used, nothing will give dim + 1
    :n_runs => 10, # Number of repeated runs
    :gpf => true, # Run GaussParticle Flow
    :advi => !true, # Run Black Box VI
    :steinvi => !true, # Run Stein VI
    :cond1 => false, # Use preconditionning on b
    :cond2 => false, # Use preconditionning on A
    :cb_val => nothing, # Callback values
    :opt => ADAGrad(0.1), # Common optimizer
    :α => 0.01,
    :σ_init => 1.0,
    :use_gpu => false,

)
ps = dict_list(exp_ps)
# @info "Will now run $(dict_list_count(exp_ps)) simulations"
# run for each dict the simulation
run_logistic_regression(ps[1])
# pmap(run_logistic_regrssion, ps)
