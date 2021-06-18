# Make sure that all packages are up to date
using DrWatson;
@quickactivate "ParticleFlow"
# using Pkg; Pkg.update()
include(srcdir("lowrank", "lowrank.jl"))

# Use parallelism
using Distributed
nthreads = 6 # Number of threads to use
nthreads = min(nthreads, Sys.CPU_THREADS - 2)
if nprocs() < nthreads
    addprocs(nthreads - nprocs() + 1) # Add the threads as workers
end

# Load all needed packages on every worker
@everywhere using DrWatson
@everywhere quickactivate("ParticleFlow")
@everywhere include(srcdir("lowrank", "lowrank.jl"))
# Create a list of parameters
exp_ps = Dict(
    :n_iters => 5000, # Number of iterations to run
    :n_runs => 10, # Number of repeated runs
    :K => 30,#,[10, 20, 30, 40],
    :n_particles => 20,
    :dof => 5.0,
    :gpf => true, # Run GaussParticle Flow
    :gf => true, # Run Gauss Flow
    :dsvi => !true, # Run Doubly Stochastic VI
    :fcs => true, # Run Factorized Structure Covariance
    :iblr => !true, # Run i Bayesian Rule
    :svgd_linear => true, # Run linear SVGD
    :svgd_rbf => true, # Run rbf SVGD
    :natmu => false, # [true, false], # Use preconditionning on b
    :seed => 42, # Seed for experiments
    :cb_val => nothing, # Callback values
    :eta => 0.01,
    :opt_det => :DimWiseRMSProp,
    :opt_stoch => :RMSProp,# :RMSProp],
    :comp_hess => :rep,
    :overwrite => false,
    :mode => :save,
)
ps = dict_list(exp_ps)
@info "Will now run $(dict_list_count(exp_ps)) simulations"
# run for each dict the simulation
# run_lowrank_target(ps[1])
# map(run_lowrank_target, ps)
pmap(run_lowrank_target, ps)
