# Make sure that all packages are up to date
using DrWatson;
@quickactivate
# using Pkg; Pkg.update()
include(srcdir("logistic", "logistic.jl"))

parallelize = false

dataset = "spam"

# Use parallelism
# if parallelize
#     using Distributed
#     nthreads = 32 # Number of threads to use
#     if nprocs() < nthreads
#         addprocs(nthreads-nprocs()+1) # Add the threads as workers
#     end
#     # Load all needed packages on every worker
#     @everywhere using DrWatson
#     @everywhere quickactivate(@__DIR__)
#     @everywhere include(srcdir("logistic", "logistic.jl"))
# end

# Create a list of parameters
exp_ps = Dict(
    :seed => 42,
    :dataset => dataset,
    :n_iters => 2001, # Number of iterations to run
    :n_particles => vcat(1:9, 10:10:99, 100:50:200),
    :k => 10, # Number of repeated runs
    :gpf => true, # Run GaussParticle Flow
    :gf => true,# Run Gauss Flow
    :dsvi => true,# Run DSVI
    :iblr => false,# Run IBLR
    :fcs => true,# Run FCS
    :natmu => false, # Natural gradient on mu
    :cb_val => nothing, # Callback values
    :eta => 0.0001, # Learning rate
    :opt_det => :Descent, # Common optimizer
    :opt_stoch => :Descent, # Common optimizer
    :comp_hess => :hess,
    :alpha => 0.1, # Prior variance
    :σ_init => 1.0, # Initial variance
    :B => 100, # Batchsize (-1 to use full batch)
    :mf => :none, # Which mean_field method should be used
)
ps = dict_list(exp_ps)
@info "Will now run $(dict_list_count(exp_ps)) simulations"
# run for each dict the simulation
run_logistic_regression(ps[10])
# pmap(run_logistic_regression, ps)
