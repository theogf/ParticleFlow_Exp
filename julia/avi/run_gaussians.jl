using DrWatson
@quickactivate
using DataFrames
using BSON
include(projectdir("avi", "gaussian_target.jl"))

exp_p = Dict(
    :n_iters => 100,
    :n_runs => 2,
    :dim => 10,
    :n_particles => 11,
    :full_cov => false,
    :gpf => true,
    :advi => true,
    :steinvi => !true,
    :cond1 => false,
    :cond2 => false,
    :seed => 42,
    :cb_val => nothing,
    :gpu => !true,
)

run_gaussian_target(exp_p)
