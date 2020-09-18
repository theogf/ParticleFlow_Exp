using DrWatson
@quickactivate
using DataFrames
using BSON
using Flux
include(srcdir("gaussian", "gaussian_target.jl"))

exp_p = Dict(
    :n_iters => 5000,
    :n_runs => 10,
    :dim => vcat(1:9, 10:10:99, 100:100:500),
    :n_particles => 11,
    :full_cov => [true, false],
    :gpf => true,
    :advi => true,
    :steinvi => true,
    :cond1 => [true, false],
    :cond2 => [true, false],
    :seed => 42,
    :cb_val => nothing,
    :opt => ADAGrad(0.1),
)

run_gaussian_target(exp_p)
