using DrWatson
@quickactivate
using DataFrames
using BSON
using Flux
include(srcdir("bnn", "gpf_bnn.jl"))

exp_p = Dict(
    :seed => 42,
    :n_epoch => 1,
    :batchsize => 128,
    :model => "LeNet",
    :dataset => "MNIST",
    :use_gpu => true,
    :start_layer => 7,
    :n_particles => 100,
    :n_iter => 200,
    :opt => ADAGrad(0.01),
    :cond1 => false,
    :cond2 => false,
    :σ_init => 1.0,
    :mf => :partial,
    :η => 1f-3, # 0.001 in Float32
    :α => 0.01, # Prior variance
)

run_gpf_bnn(exp_p)
