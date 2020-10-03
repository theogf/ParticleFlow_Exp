using DrWatson
@quickactivate
using Pkg; Pkg.update()

using DataFrames
using BSON
using Flux
include(srcdir("bnn", "swag.jl"))

exp_ps = Dict(
    :n_epoch => 100,
    :batchsize => 128,
    :model => "LeNet",
    :dataset => "MNIST",
    :use_gpu => true,
    :start_layer => [1, 7, 8, 9],
    :seed => 42,
    :n_period => 10,
    :η => 1f-3, # 0.001 in Float32
    :α => 0.01,
)

ps = dict_list(exp_ps)
@info "Will now run $(dict_list_count(exp_ps)) simulations"

for (i, p) in enumerate(ps)
    @info "Running dict $(i)/$(length(ps)) : $(savename(p))"
    run_SWAG(p)
end
