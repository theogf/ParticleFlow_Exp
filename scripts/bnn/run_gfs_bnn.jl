using DrWatson
@quickactivate("ParticleFlow")
# using Pkg; Pkg.update()

using DataFrames
using BSON
using Flux
using CUDA
include(srcdir("bnn", "bnn.jl"))

exp_ps = Dict(
    :seed => 42,
    :batchsize => 128,
    :model => "BNN",
    :n_hidden => [100, 200, 400, 800],
    :activation => [:relu, :tanh],
    :dataset => "MNIST",
    :use_gpu => true,
    :alg => :dsvi,
    # :alg => [:gpf, :gf, :fcs, :dsvi, :svgd_linear, :svgd_rbf],
    :L => [10, 50, 100],
    :n_iter => 5001,
    :opt_det => :DimWiseRMSProp,
    :opt_stoch => :RMSProp,
    :eta => 1f-2,
    :natmu => false,
    :σ_init => 1.0,
    :mf => :full, #[:full, :partial, :none],
    :α => [0.001, 0.01, 0.1, 1.0, 10.0] # Prior variance
)

ps = dict_list(exp_ps)
@info "Will now run $(dict_list_count(exp_ps)) simulations"

run_bnn(ps[1])

# for (i, p) in enumerate(ps)
#     @info "Running dict $(i)/$(length(ps)) : $(savename(p))"
#     GC.gc(true)
#     CUDA.reclaim()
#     run_gpf_bnn(p)
# end
