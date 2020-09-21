using DrWatson
@quickactivate
include(srcdir("gp", "true_var_gp.jl"))

exp_p = Dict(
    :seed => 42,
    :dataset => "ionosphere",
    :nSamples => 10_000,
    :nBurnin => 200,
)

run_true_gp(exp_p)
