using DrWatson
@quickactivate
using DataFrames
using BSON
using Flux
include(srcdir("gaussian", "gaussian_target.jl"))

exp_p = Dict(
    :n_epoch => 10,
    :batchsize => 128,
    :model => "LeNet",
    :dataset => "MNIST",
    :use_gpu => true,
    :layers => 7:end,
    :seed => 42,
    :n_period => 10
    :η => 1f-3 # 0.001 in Float32
    :α => 0.01
)

run_gaussian_target(exp_p)
