using DrWatson
@quickactivate
include(projectdir("analysis", "post_process.jl"))
include(srcdir("utils", "bnn.jl"))
pyplot()
using Flux
using StatsBase, LinearAlgebra
## Load data and filter it
dataset = "MNIST"
model = "LeNet"
batchsize = 128
n_epoch = 100
n_period = 10
η = 0.001
cond1 = false
cond2 = false
start_layer = 9
K = 30
## Load SWAG data
swag_res = collect_results(datadir("results", "bnn", dataset, "SWAG", savename(@dict batchsize n_epoch n_period η start_layer)))
thinning = 10
res = ([vcat(vec.(x)...) for x in swag_res.parameters[1:thinning:end]])
using Plots
SWA_sqrt_diag = Diagonal(StatsBase.std(res))
SWA = mean(res[end-K+1:end])
SWA_D = reduce(hcat, res[end-K+1:end] .- Ref(SWA))

## Load model and data
modelfile = projectdir("bnn_models", model, "model.bson")
m = BSON.load(modelfile)[:model]
fixed_m = m[1:(start_layer-1)]
opt_m = m[start_layer:end]
opt_θ, opt_re = Flux.destructure(opt_m)
n_θ = length(opt_θ)
function nn_forward(xs, θ)
    opt_m = opt_re(θ)
    nn = Chain(fixed_m, opt_m)
    return nn(xs)
end
## Loading data
train_loader, test_loader = get_data(dataset, 10_000);
X_test, y_test = first(test_loader)
opt_pred = Flux.softmax(nn_forward(X_test, opt_θ))

## Create predictions using SWAG
n_MC = 100

preds = []
@progress for i in 1:n_MC
    θ = SWA + SWA_sqrt_diag / sqrt(2f0) * randn(Float32, n_θ) + SWA_D / sqrt(2f0 * (K - 1)) * randn(Float32, K)
    pred = nn_forward(X_test, θ)
    push!(preds, Flux.softmax(pred))
end
SWAG_preds = mean(preds)
function max_ps_ids(X)
    maxs = findmax.(eachcol(X))
    return ps, ids = first.(maxs), last.(maxs)
end

opt_ps, opt_ids = max_ps_ids(opt_pred)
bins = range(0, 1, length = 20)
StatsBase.fit(StatsBase.Histogram, opt_ps, bins)

## Predictions with GPF
n_particles = 200
mf = :none
n_iter = 5000
gpf_res = collect_results(datadir("results", "bnn", dataset, "GPF", @savename n_particles n_iter batchsize mf cond1 cond2))
names(gpf_res)
particles = first(gpf_res.particles[gpf_res.i .== n_iter])
preds = []
@progress for θ in eachcol(particles)
    pred = nn_forward(X_test, θ)
    push!(preds, Flux.softmax(pred))
end
gpf_preds = mean(preds)




##
p_μ = plot(title = "Convergence Mean", xlabel = "Time [s]", ylabel =L"\|\mu - \mu_{true}\|", xaxis=:log)
p_Σ = plot(title = "Convergence Covariance", xlabel = "Time [s]", ylabel =L"\|\Sigma - \Sigma_{true}\|", xaxis=:log)
for (i, alg) in enumerate(algs)
    @info "Processing $(alg)"
    t_alg = Symbol("t_", alg); t_var_alg = Symbol("t_var_", alg)
    m_alg = Symbol("m_", alg); m_var_alg = Symbol("m_var_", alg)
    v_alg = Symbol("v_", alg); v_var_alg = Symbol("v_var_", alg)
    @eval $(t_alg), $(t_var_alg) = process_time(first(res.$(alg)))
    @eval $(m_alg), $(m_var_alg) = process_means(first(res.$(alg)), truth.m)
    @eval $(v_alg), $(v_var_alg) = process_fullcovs(first(res.$(alg)), vec(truth.C.L * truth.C.U))
    @eval plot!(p_μ, $(t_alg), $(m_alg), ribbon = sqrt.($(m_var_alg)), label = $(labels[alg]), color = colors[$i])
    @eval plot!(p_Σ, $(t_alg), $(v_alg), ribbon = sqrt.($(v_var_alg)), label = $(labels[alg]), color = colors[$i])
end
display(plot(p_μ, p_Σ))
