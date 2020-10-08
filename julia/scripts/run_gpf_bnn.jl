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
    :start_layer => [7, 8, 9],
    :n_particles => [10, 50, 100],
    :n_iter => 5000,
    :opt => Flux.Optimise.Optimiser(ClipNorm(10), Descent(0.1)),
    :cond1 => false,
    :cond2 => false,
    :σ_init => 1.0,
    :mf => [:full, :partial, :none],
    :α => [100.0, 10.0, 1.0, 0.1], # Prior variance
)

ps = dict_list(exp_ps)
@info "Will now run $(dict_list_count(exp_ps)) simulations"

# run_gpf_bnn(ps[end])

for (i, p) in enumerate(ps)
    @info "Running dict $(i)/$(length(ps)) : $(savename(p))"
    run_gpf_bnn(p)
end
