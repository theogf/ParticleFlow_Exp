# Make sure that all packages are up to date
using DrWatson
@quickactivate
using Pkg; Pkg.update()

#Use parallelism
using Distributed
nthreads = 6 # Number of threads to use
if nprocs() < nthreads
    addprocs(nthreads-nprocs()+1) # Add the threads as workers
end

# Load all needed packages on every worker
include(srcdir("gp", "gp_gdp.jl"))
@everywhere using DrWatson
@everywhere quickactivate(@__DIR__)
@everywhere include(srcdir("gp", "gp_gdp.jl"))

dataset = "ionosphere"
preload(dataset, "gp")


exp_p = Dict(
    :seed => 42, # Random seed
    :dataset => dataset, # Name of the dataset
    :n_particles => vcat(1:9, 10:10:100),#, 100:50:400), # Number of particles used
    :n_iters => 2000, # Total number of iterations
    :n_runs => 10, # Number of repeated runs
    :cond1 => false, # Preconditionning on b
    :cond2 => false, # Preconditionning on A
    :σ_init => 1.0, # Initial std dev
    :opt => Flux.Optimise.Optimiser(ClipNorm(10), Descent(0.1)), # Optimizer used
)

ps = dict_list(exp_p) # Create list of parameters
@info "Preparing to run $(dict_list_count(exp_p)) simulations"

# Running simulations
# run_gp_gdp(ps[1])
map(run_gp_gdp, ps[2:end])
