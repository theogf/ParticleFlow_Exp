# Make sure that all packages are up to date
using DrWatson;
@quickactivate
using Pkg; Pkg.update()

# Use parallelism
using Distributed
nthreads = 32 # Number of threads to use
if nprocs() < nthreads
    addprocs(nthreads-nprocs()+1) # Add the threads as workers
end

# Load all needed packages on every worker
@everywhere using DrWatson
@everywhere quickactivate(@__DIR__)
@everywhere include(srcdir("linear", "linear.jl"))
include(srcdir("linear", "linear.jl"))


dataset = "swarm_flocking"

preload(dataset, "linear")

# Create a list of parameters
exp_ps = Dict(
    :seed => 42,
    :dataset => "swarm_flocking",
    :n_iters => 2000, # Number of iterations to run
    :n_particles => vcat(1:9, 10:10:99, 100:50:200),
    :n_runs => 10, # Number of repeated runs
    :gpf => true, # Run GaussParticle Flow
    :advi => @onlyif(:mf !== :none, true), # Run Black Box VI
    :steinvi => @onlyif(:mf == :none, true), # Run Stein VI
    :cond1 => false, # Use preconditionning on b
    :cond2 => false, # Use preconditionning on A
    :cb_val => nothing, # Callback values
    :opt => ADAGrad(0.1), # Common optimizer
    :α => 0.01, # Prior variance
    :σ_init => 1.0, # Initial variance
    :B => 200, # Batchsize
    :use_gpu => false, # Use of the GPU (tends do be inefficient)
    :mf => [:full, :partial, :none], # Which mean_field method should be used

)
ps = dict_list(exp_ps)
@info "Will now run $(dict_list_count(exp_ps)) simulations"
# run for each dict the simulation
# run_logistic_regression(ps[2])
pmap(run_logistic_regression, ps)
