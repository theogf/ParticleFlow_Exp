using DrWatson
@quickactivate
using Pkg; Pkg.update()

using CUDA

include(srcdir("bnn", "swag.jl"))

exp_ps = Dict(
    :n_epoch => 50,
    :batchsize => 128,
    :model => "LeNet",
    :dataset => "MNIST",
    :use_gpu => true,
    :start_layer => 7,#[1, 7, 8, 9],
    :seed => 42,
    :n_period => 10,
    :η => 1f-2,#[1f-1, 5f-2, 1f-2], # 0.001 in Float32
    :α => [0.01, 0.05, 0.1, 1.0, 5.0, 10, 50, 100],
)

ps = dict_list(exp_ps)
@info "Will now run $(dict_list_count(exp_ps)) simulations"

for (i, p) in enumerate(ps)
    @info "Running dict $(i)/$(length(ps)) : $(savename(p))"
    run_SWAG(p)
end
