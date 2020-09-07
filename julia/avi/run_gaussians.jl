using DrWatson
@quickactivate
using DataFrames
using BSON
include(projectdir("avi", "gaussian_target.jl"))

exp_p = Dict(
    :n_iters => 100,
    :n_runs => 10,
    :dim => 10,
    :n_particles => 11,
    :full_cov => false,
    :gpf => true,
    :advi => true,
    :steinvi => true,
    :cond1 => false,
    :cond2 => false,
    :seed => 42,
)

run_gaussian_target(exp_p)
