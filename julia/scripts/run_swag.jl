using DrWatson
@quickactivate
using DataFrames
using BSON
using Flux
include(srcdir("bnn", "swag.jl"))

exp_p = Dict(
    :n_epoch => 1,
    :batchsize => 128,
    :model => "LeNet",
    :dataset => "MNIST",
    :use_gpu => true,
    :start_layer => 7,
    :seed => 42,
    :n_period => 10,
    :η => 1f-3, # 0.001 in Float32
    :α => 0.01,
)

run_SWAG(exp_p)
