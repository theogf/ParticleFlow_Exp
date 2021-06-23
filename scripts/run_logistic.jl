# Make sure that all packages are up to date
using DrWatson;
@quickactivate "ParticleFlow"
# using Pkg; Pkg.update()
include(srcdir("logistic", "logistic.jl"))

dataset = ["spam", "mushroom", "ionosphere", "krkp"]

# Use parallelism
using Distributed
nthreads = 60 # Number of threads to use
nthreads = min(nthreads, Sys.CPU_THREADS - 2)
if nprocs() < nthreads
    addprocs(nthreads - nprocs() + 1) # Add the threads as workers
end
# Load all needed packages on every worker
@everywhere using DrWatson
@everywhere @quickactivate "ParticleFlow"
@everywhere include(srcdir("logistic", "logistic.jl"))

# Create a list of parameters
exp_ps = Dict(
    :seed => 42,
    :dataset => dataset,
    :n_iters => 5000, # Number of iterations to run
    :n_particles => [2, 5, 100],
    :p => 100,
    :k => 10, # Number of repeated runs
    :gpf => !true, # Run GaussParticle Flow
    :gf => true,# Run Gauss Flow
    :dsvi => true,# Run DSVI
    :iblr => !true,# Run IBLR
    :fcs => true,# Run FCS
    :svgd_linear => false,# Run linear SVGD
    :svgd_rbf => false,# Run RBF SVGD
    :natmu => false, #[true, false], # Natural gradient on mu
    :cb_val => nothing, # Callback values
    :eta => 0.0001, # Learning rate
    :opt_det => :RMSProp, # Common optimizer
    :opt_stoch => :RMSProp, # Common optimizer
    :comp_hess => :rep,
    :alpha => 0.1, # Prior variance
    :σ_init => 1.0, # Initial variance
    :B => [-1, 100], # Batchsize (-1 to use full batch)
    :mf => [:full, :none], # Which mean_field method should be used,
    :unsafe => true,
)
ps = dict_list(exp_ps)
@info "Will now run $(dict_list_count(exp_ps)) simulations"
# run for each dict the simulation
run_logistic_regression(ps[1])
# map(run_logistic_regression, ps)
pmap(run_logistic_regression, ps)
