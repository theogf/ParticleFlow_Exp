using DrWatson
@quickactivate
include(srcdir("gp", "gp_gpf.jl"))

exp_p = Dict(
    :seed => 42,
    :dataset => "ionosphere",
    :n_particles => 10,
    :n_iters => 1000,
    :n_runs => 10,
    :cond1 => false,
    :cond2 => false,
    :σ_init => 1.0,
    :opt => ADAGrad(0.01),
)


run_gp_gpf(exp_p)
