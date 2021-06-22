cd(@__DIR__)
using AdvancedVI, Bijectors
using Turing, Flux, Plots, Random;
include("../src/train_model.jl")
include("../src/utils/tools.jl")
# using ONNX
using BSON
const AVI = AdvancedVI
using Parameters: @with_kw
using MLDatasets
using ReverseDiff
using ValueHistories
using CUDA
device = gpu
AVI.setadbackend(:reversediff)
AVI.setadbackend(:zygote)
model = "squeezenet1.1"
model = "mobilenetv2-1.0"
model = "lenet_mnist"
model_dir = joinpath(
    "/home",
    "theo",
    "experiments",
    "ParticleFlow",
    "julia",
    "pretrained_models",
    model,
)
# ONNX.load_model(joinpath(model_dir, model * ".onnx"))
# weights = ONNX.load_weights(joinpath(model_dir, "weights.bson"))
# model = include(joinpath(model_dir, "model.jl"))
m = BSON.load(joinpath(model_dir, "model.bson"))[:model]
middle = 7
convm = m[1:middle]
densem = m[middle+1:end]
dense_θ, dense_re = Flux.destructure(densem)
convm = convm |> device
n_θ = length(dense_θ)
densem_sizes = broadcast(x-> length(x.W) + length(x.b), densem)
nn_indices = vcat(0, cumsum(densem_sizes))
# K = 1000
# nn_indices = ceil.(Int, collect(range(0, n_θ, length = K)))

function nn_forward(xs, nn_params::AbstractVector)
    densem = dense_re(nn_params)
    nn = Chain(convm, densem)
    return nn(xs)
end;


## Create model


# Create a regularization term and a Gaussain prior variance term.
alpha = 0.09
sig = sqrt(1.0 / alpha)
args = Args()

train_loader, test_loader = get_data(args)
x, y = first(train_loader)

# Put prior on GPU
Flux.@functor TuringDiagMvNormal
prior_θ = TuringDiagMvNormal(zeros(n_θ), sig .* ones(n_θ)) |> device
N_batch = length(train_loader)
function meta_logjoint(dummy)
    s = rand(1:N_batch)
    xs, ys = Random.nth(train_loader, s)
    xs = xs |> device
    ys = ys |> device
    return function logjoint(θ)
        logprior = logpdf(prior_θ, θ)
        pred = nn_forward(xs, θ)
        loglike = Flux.Losses.logitcrossentropy(pred, ys) * N_batch
        return logprior - loglike
    end
end

function evaluate_acc(x)
    pred = nn_forward(xs)
end

function cb_val(h, i, q, hp)
    s = rand(1:length(test_loader))
    xt, yt = Random.nth(test_loader, s) |> device
    ŷ = mean(eachcol(q.dist.x)) do θ
        pred = softmax(nn_forward(xt, θ))
    end
    @show ll = Flux.Losses.crossentropy(ŷ, yt)
    @show acc = mean(yt[findmax(ŷ, dims = 1)[2]])
    push!(h, :ll, i, ll)
    push!(h, :acc, i, acc)
end

GC.gc(true)
CUDA.reclaim()


norun = Dict(:run => false)
general_p = Dict(:hyper_params => [], :hp_optimizer => nothing, :n_dim => n_θ, :gpu => device == gpu)
n_particles = 100
n_iters = 1000
cond1 = false
cond2 = false
opt = ADAGrad(0.1)
σ² = 0.1
init_particles = rand(MvNormal(dense_θ, Float32(σ²)), n_particles)
gflow_p = Dict(
    :run => true,
    :n_particles => n_particles,
    :max_iters => n_iters,
    :cond1 => cond1,
    :cond2 => cond2,
    :opt => deepcopy(opt),
    :callback => wrap_heavy_cb(;cb_val = cb_val, path = joinpath(@__DIR__, "tests_models")),
    :cb_val => nothing,
    :init => init_particles,
    :mf => nn_indices,
    :gpu => device == gpu
)

## running


g_h, _, _ = train_model(meta_logjoint, general_p, gflow_p, norun, norun)

# @profiler train_model(meta_logjoint, general_p, gflow_p, norun, norun)
# CUDA.@time train_model(meta_logjoint, general_p, gflow_p, norun, norun)
# last(g_h[:sig])[2]
# last(g_h[:mu])[2]
# last(g_h[:indices])[2]
##
# CUDA.@profile train_model(meta_logjoint, general_p, gflow_p, norun, norun)
