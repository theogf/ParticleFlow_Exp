using DrWatson
@quickactivate
using DataFrames
using BSON
include(projectdir("avi", "classification_svgp.jl"))

exp_p = Dict(
    :n_iters => 10,
    :n_runs => 10,
    :n_particles => 11,
    :data => "toydata",
    :n_batch => 30,
    :n_ind_points => 20,
    :t_gate => 0,
    :full_cov => false,
    :gpf => true,
    :cond1 => false,
    :cond2 => false,
    :seed => 42,
)

run_svgp_bin(exp_p)
