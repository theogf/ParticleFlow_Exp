# Make sure that all packages are up to date
using DrWatson;
@quickactivate
# using Pkg; Pkg.update()
include(srcdir("gaussian", "gaussian_target.jl"))

# Use parallelism
using Distributed
nthreads = 3 # Number of threads to use
nthreads = min(nthreads, Sys.CPU_THREADS - 2)
if nprocs() < nthreads
    addprocs(nthreads - nprocs() + 1) # Add the threads as workers
end

# Load all needed packages on every worker
@everywhere using DrWatson
@everywhere @quickactivate
@everywhere include(srcdir("gaussian", "gaussian_target.jl"))
# Create a list of parameters
exp_ps = Dict(
    :n_iters => 30000, # Number of iterations to run
    :n_runs => 10, # Number of repeated runs
    :n_dim => 20, #[50, 100], #[5, 10, 20], #50, 100], # Dimension of the target
    :n_particles => 0,#, 10, 20, 50, 100], # Number of particles used, nothing will give dim + 1
    :cond => [1, 10, 100],
    :gpf => !true, # Run GaussParticle Flow
    :gf => !true, # Run Gauss Flow
    :dsvi => !true, # Run Doubly Stochastic VI
    :fcs => !true, # Run Factorized Structure Covariance
    :iblr => !true, # Run i Bayesian Rule
    :svgd_linear => false, # Run linear SVGD
    :svgd_rbf => true,
    :natmu => false, # Use preconditionning on b
    :seed => 42, # Seed for experiments
    :cb_val => nothing, # Callback values
    :eta => 0.01,
    :opt_det => :Descent,
    :opt_stoch => :RMSProp,
    :comp_hess => :rep,
    :overwrite => :true,
    :mode => :save,
)
ps = dict_list(exp_ps)
@info "Will now run $(dict_list_count(exp_ps)) simulations"
# run for each dict the simulation
# run_gaussian_target(ps[1])
# map(run_gaussian_target, ps)
pmap(run_gaussian_target, ps)
