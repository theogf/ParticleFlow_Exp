using DrWatson
@quickactivate
using Pkg; Pkg.update()

using DataFrames
using BSON
using Flux
using CUDA
include(srcdir("bnn", "advi_bnn.jl"))

exp_ps = Dict(
    :seed => 42,
    :batchsize => 128,
    :model => "LeNet",
    :dataset => "MNIST",
    :use_gpu => true,
    :start_layer => 7,
    :n_particles => [10, 50],
    :n_iter => 5001,
    :opt => Flux.Optimise.Optimiser(ClipNorm(10), Descent(0.1)),
    :α => [0.05, 0.01] # Prior variance
)

ps = dict_list(exp_ps)
@info "Will now run $(dict_list_count(exp_ps)) simulations"

# run_gpf_bnn(ps[end])

for (i, p) in enumerate(ps)
    @info "Running dict $(i)/$(length(ps)) : $(savename(p))"
    GC.gc(true)
    CUDA.reclaim()
    run_advi_bnn(p)
end