using DrWatson
@quickactivate
using DataFrames
using BSON
using Flux
include(srcdir("gaussian", "gaussian_target.jl"))

exp_p = Dict(
    :n_iters => 1000,
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
    :cb_val => nothing,
    :opt => ADAGrad(0.1),
)

run_gaussian_target(exp_p)
