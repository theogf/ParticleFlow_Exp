using DrWatson
@quickactivate
include(projectdir("analysis", "post_process.jl"))
include(srcdir("utils", "bnn.jl"))
pyplot()
using Flux
using StatsBase, LinearAlgebra
using MLDataUtils
using ProgressMeter
use_gpu = true
dev = use_gpu ? gpu : cpu
## Load data and filter it
dataset = "MNIST"
model = "LeNet"
batchsize = 128
n_epoch = 100
n_period = 10
η = 0.01
α = 0.1
cond1 = false
cond2 = false
start_layer = 7
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
fixed_m = m[1:(start_layer-1)] |> dev
opt_m = m[start_layer:end]
opt_θ, opt_re = Flux.destructure(opt_m)
n_θ = length(opt_θ)
function nn_forward(xs, θ)
    opt_m = opt_re(θ) |> dev
    nn = Chain(fixed_m, opt_m)
    return nn(xs)
end
## Loading data
train_loader, test_loader = get_data(dataset, 10_000);
X_test, y_test = first(test_loader)
X_test = X_test |> dev
N20 = size(X_test, 4) ÷ 20

opt_pred = cpu(Flux.softmax(nn_forward(X_test, opt_θ)))

function max_ps_ids(X)
    maxs = findmax.(eachcol(X))
    return ps, ids = first.(maxs), last.(maxs)
end
opt_ps, opt_ids = max_ps_ids(opt_pred)
function conf_and_acc(preds)
    conf_and_acc(max_ps_ids(preds)...)
end
function conf_and_acc(ps, ids)
    s = sortperm(ps)
    bins = [s[i*N20+1:(i+1)*N20] for i in 0:19]
    conf = mean.(getindex.(Ref(ps), bins))
    acc = [mean(ids[bins[i]] .== Flux.onecold(y_test)[bins[i]]) for i in 1:20]
    return conf, acc
end


## Create predictions using SWAG
n_MC = 100

preds = []
@showprogress for i in 1:n_MC
    θ = SWA + SWA_sqrt_diag / sqrt(2f0) * randn(Float32, n_θ) + SWA_D / sqrt(2f0 * (K - 1)) * randn(Float32, K)
    pred = nn_forward(X_test, θ)
    push!(preds, cpu(Flux.softmax(pred)))
end
SWAG_preds = mean(preds)
SWAG_nll = Flux.Losses.crossentropy(SWAG_preds, y_test)
SWAG_acc = mean(Flux.onecold(SWAG_preds) .== Flux.onecold(y_test))

## Predictions with GPF and ADVI + Plotting
accs = Dict()
accs[:gpf] = Dict(); accs[:advi] = Dict()
mfs = [:full, :partial, :none]
for mf in mfs
    accs[:gpf][mf] = Dict()
end
nlls = Dict()
nlls[:gpf] = Dict(); nlls[:advi] = Dict()
for mf in mfs
    nlls[:gpf][mf] = Dict()
end
αs = [0.1, 1, 5, 10, 100]
for n_particles in [10, 50, 100],
    α in αs
    @show α, n_particles
    # Deal with GPF
    gpf_conf = Dict()
    gpf_acc = Dict()
    for mf in mfs
        n_iter = 5001
        gpf_res = collect_results(datadir("results", "bnn", dataset, "GPF_LeNet", @savename start_layer n_particles α n_iter batchsize mf cond1 cond2))
        particles = first(gpf_res.particles[gpf_res.i .== n_iter-1])
        preds = []
        @showprogress for θ in eachcol(particles)
            pred = nn_forward(X_test, θ)
            push!(preds, cpu(Flux.softmax(pred)))
        end
        gpf_preds = mean(preds)
        nlls[:gpf][mf][(α, n_particles)] = Flux.Losses.crossentropy(gpf_preds, y_test)
        accs[:gpf][mf][(α, n_particles)] = mean(Flux.onecold(gpf_preds) .== Flux.onecold(y_test))
        gpf_conf[mf], gpf_acc[mf] = conf_and_acc(gpf_preds)
    end

    # Deal with ADVI
    n_iter = 5001
    advi_res = collect_results(datadir("results", "bnn", dataset, "ADVI_LeNet", @savename start_layer n_particles α n_iter batchsize))
    q = first(advi_res.q[advi_res.i .== n_iter-1])
    preds = []
    @showprogress for θ in eachcol(rand(q, n_particles))
        pred = nn_forward(X_test, θ)
        push!(preds, cpu(Flux.softmax(pred)))
    end
    advi_preds = mean(preds)

    nlls[:advi][(α, n_particles)] = Flux.Losses.crossentropy(advi_preds, y_test)
    accs[:advi][(α, n_particles)] = mean(Flux.onecold(advi_preds) .== Flux.onecold(y_test))

    ## Plotting of confidence histogram

    opt_conf, opt_acc = conf_and_acc(opt_pred)
    advi_conf, advi_acc = conf_and_acc(advi_preds)
    swag_conf, swag_acc = conf_and_acc(SWAG_preds)
    p = plot(title = "LeNet - α = $α", xaxis = "Confidence (max prob)", yaxis = "Accuracy")
    plot!([0.5, 1], identity, linestyle = :dash, color = :black, label = "")
    plot!(opt_conf, opt_acc, marker = :o, label = "ML")
    for mf in mfs
        plot!(gpf_conf[mf], gpf_acc[mf], marker = :o, label = "GPF - $mf - $(n_particles)")
    end
    plot!(advi_conf, advi_acc, marker = :o, label = "GVA - $(n_particles)")
    plot!(swag_conf, swag_acc, marker = :o, label = "SWAG")
    savefig(plotsdir("bnn", savename("confidencelenet", @dict(α, n_particles), "png")))
    display(p)
end
## Collect nll and accs
n_ps = [10, 50, 100]
nll = Dict()
nll[:gpf] = Dict()
for mf in mfs
    nll[:gpf][mf] = Vector(undef, length(n_ps))
end
nll[:advi] = Vector(undef, length(n_ps))

acc = Dict()
acc[:gpf] = Dict()
for mf in mfs
    acc[:gpf][mf] = Vector(undef, length(n_ps))
end
acc[:advi] = Vector(undef, length(n_ps))

for (i, n_particles) in enumerate(n_ps)
    for mf in mfs
        nll[:gpf][mf][i] = [nlls[:gpf][mf][(α, n_particles)] for α in αs]
    end
    nll[:advi][i] = [nlls[:advi][(α, n_particles)] for α in αs]
    for mf in mfs
        acc[:gpf][mf][i] = [accs[:gpf][mf][(α, n_particles)] for α in αs]
    end
    acc[:advi][i] = [accs[:advi][(α, n_particles)] for α in αs]
end
for (i, n_particles) in enumerate(n_ps)
    p = plot(xlabel = "α", xaxis = :log, yaxis = "Neg. Log-Likelihood", title = "# particles : $n_particles")
    for mf in mfs
        plot!(αs, nll[:gpf][mf][i], label = "GPF - $mf")
    end
    plot!(αs, nll[:advi][i], label = "GVA")
    hline!([SWAG_nll], label = "SWAG")
    display(p)

    p = plot(xlabel = "α", xaxis = :log, yaxis = "Accuracy", title = "# particles : $n_particles")
    for mf in mfs
        plot!(αs, acc[:gpf][mf][i], label = "GPF - $mf")
    end
    plot!(αs, acc[:advi][i], label = "GVA")
    hline!([SWAG_acc], label = "SWAG")
    display(p)

end


##
# p_μ = plot(title = "Convergence Mean", xlabel = "Time [s]", ylabel =L"\|\mu - \mu_{true}\|", xaxis=:log)
# p_Σ = plot(title = "Convergence Covariance", xlabel = "Time [s]", ylabel =L"\|\Sigma - \Sigma_{true}\|", xaxis=:log)
# for (i, alg) in enumerate(algs)
#     @info "Processing $(alg)"
#     t_alg = Symbol("t_", alg); t_var_alg = Symbol("t_var_", alg)
#     m_alg = Symbol("m_", alg); m_var_alg = Symbol("m_var_", alg)
#     v_alg = Symbol("v_", alg); v_var_alg = Symbol("v_var_", alg)
#     @eval $(t_alg), $(t_var_alg) = process_time(first(res.$(alg)))
#     @eval $(m_alg), $(m_var_alg) = process_means(first(res.$(alg)), truth.m)
#     @eval $(v_alg), $(v_var_alg) = process_fullcovs(first(res.$(alg)), vec(truth.C.L * truth.C.U))
#     @eval plot!(p_μ, $(t_alg), $(m_alg), ribbon = sqrt.($(m_var_alg)), label = $(labels[alg]), color = colors[$i])
#     @eval plot!(p_Σ, $(t_alg), $(v_alg), ribbon = sqrt.($(v_var_alg)), label = $(labels[alg]), color = colors[$i])
# end
# display(plot(p_μ, p_Σ))
