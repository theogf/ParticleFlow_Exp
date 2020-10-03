using DrWatson
@quickactivate
using Pkg; Pkg.update()

using DataFrames
using BSON
using Flux
include(srcdir("bnn", "gpf_bnn.jl"))

exp_ps = Dict(
    :seed => 42,
    :batchsize => 128,
    :model => "LeNet",
    :dataset => "MNIST",
    :use_gpu => true,
    :start_layer => [1, 7, 8, 9],
    :n_particles => [10, 50, 100, 200],
    :n_iter => 5000,
    :opt => ADAGrad(0.01),
    :cond1 => false,
    :cond2 => false,
    :σ_init => 1.0,
    :mf => [:full, :partial, :none],
    :α => 0.01, # Prior variance
)

ps = dict_list(exp_ps)
@info "Will now run $(dict_list_count(exp_ps)) simulations"

for (i, p) in enumerate(ps)
    @info "Running dict $(i)/$(length(ps)) : $(savename(p))"
    run_gpf_bnn(p)
end
