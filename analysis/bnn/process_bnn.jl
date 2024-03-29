using DrWatson
@quickactivate
include(projectdir("analysis", "post_process.jl"))
include(srcdir("utils", "bnn.jl"))
include("utils_process_bnn.jl")
pyplot()
using Flux
using StatsBase, LinearAlgebra
using MLDataUtils
using DataFrames
using Latexify
using ProgressMeter
using Plots
using CUDA
use_gpu = true
dev = use_gpu ? gpu : cpu
if use_gpu
    GC.gc(true)
    CUDA.reclaim()
end
## Load data and filter it
dataset = "MNIST"
model = "BNN"
n_hidden = 200
activation = :tanh
exp_params = Dict(
    :batchsize => 128,
    :n_epoch => 50,
    :n_period => 10,
    :eta => 0.01,
    :α => 1.0,
    :natmu => false,
    :L => 5,
    :n_iter => 5001,
    :opt_det => :DimWiseRMSProp,
    :opt_stoch => :RMSProp,
    :σ_init => 1.0,
)
n_MC = 100

## Load model and data
model = "BNN_" * @savename(activation, n_hidden)
modelfile = datadir("exp_raw", "bnn", "models", dataset, @savename(activation, n_hidden) * ".bson")
m = BSON.load(modelfile)[:nn]
θ, re = Flux.destructure(m)
n_θ = length(θ)
function nn_forward(xs, θ)
    nn = re(θ) |> dev
    return nn(reshape(xs, 28 * 28, :))
end
## Loading data
train_loader, test_loader = get_data(dataset, 10_000);
X_test, y_test = first(test_loader)
X_test = X_test |> dev
N20 = size(X_test, 3) ÷ 20

opt_pred = cpu(Flux.softmax(nn_forward(X_test, θ)))
opt_ps, opt_ids = max_ps_ids(opt_pred)

bnn_algs = [
    :gpf,
    :gf,
    :dsvi,
    :svgd_rbf,
    :swag,
    :elrgvi,
    # :slang,
]

algs_to_mf = Dict(
    :gpf => [:none, :full, :partial],
    :gf => [:none, :full, :partial],
    :dsvi => [:full],
    :svgd_linear => [:none],
    :svgd_rbf => [:none],
    :swag => [:none],
    :elrgvi => [:none],
    :slang => [:none],
)

## Load the results
nlls = Dict()
accs = Dict()
confs = Dict()
conf_accs = Dict()
n_iter = 5001

for alg in bnn_algs # Loop over every algorithm
    alg_dir = datadir("results", "bnn", dataset, model, string(alg))
    nlls[alg] = Dict()
    accs[alg] = Dict()
    confs[alg] = Dict()
    conf_accs[alg] = Dict()
    for mf in algs_to_mf[alg]
        nlls[alg][mf], accs[alg][mf], confs[alg][mf], conf_accs[alg][mf] = extract_info(Val(alg), alg_dir, mf, exp_params)
        if use_gpu
            GC.gc(true)
            CUDA.reclaim()
        end
    end
end


## Plotting of confidence histogram
plt_none = plot(;xflip = false, legendfontsize = 13.5, legend = :topleft, title = "L = $L", xlabel = "Confidence (max prob)", ylabel = "Accuracy")
plt_mf = plot(;xflip = false, legendfontsize = 13.5, legend = :topleft, title = "Mean-Field", xlabel = "Confidence (max prob)", ylabel = "Accuracy")
# xticks!(logxvals, string.(xvals))
msw = 0.5
ms = 8.0
lw = 0.0 # 5.0
plot!(plt_none, [0.0, 1.0], identity, linestyle = :dash, color = :black, label = "")
plot!(plt_mf, [0.0, 1.0], identity, linestyle = :dash, color = :black, label = "")
# plot!(opt_conf,   opt_acc, marker = :o, label = "ML", msw = msw, color = colors[7])
@unpack eta, L = exp_params
alpha= exp_params[:α]
for alg in bnn_algs
    for mf in algs_to_mf[alg]
        if !isnothing(confs[alg][mf])
            if mf == :full
                scatter!(plt_mf, last(confs[alg][mf]), last(conf_accs[alg][mf]), marker = alg_markers[alg], label = alg_lab[alg], color = alg_col[alg], linestyle = alg_mf_line[mf], msw = msw, linewidth = lw, ms = ms)
            else
                scatter!(plt_none, last(confs[alg][mf]), last(conf_accs[alg][mf]), marker = alg_markers[alg], label = ece_labs[(alg, mf)], color = alg_col[alg], linestyle = alg_mf_line[mf], msw = msw, linewidth = lw, ms = ms)
            end
        end
    end
end
plot_dir = plotsdir("bnn")
mkpath(plot_dir)
savefig(plt_none, joinpath(plot_dir, savename("confidence_bnn_none_", @dict(n_hidden, activation, alpha, eta, L), "png")))
savefig(plt_none, joinpath(plot_dir, savename("confidence_bnn_none_", @dict(n_hidden, activation, alpha, eta, L), "eps")))
savefig(plt_mf, joinpath(plot_dir, savename("confidence_bnn_full_", @dict(n_hidden, activation, alpha, eta, L), "png")))
savefig(plt_mf, joinpath(plot_dir, savename("confidence_bnn_full_", @dict(n_hidden, activation, alpha, eta, L), "eps")))
display(plot(plt_none, plt_mf))


## Work with the NLL and ACC
alg_labels = String[]
alg_nll = Float64[]
alg_acc = Float64[]
alg_name = String[]
alg_type = String[]
eces = Float64[]
for alg in bnn_algs
    for mf in algs_to_mf[alg]
        if !isnothing(nlls[alg][mf])
            push!(alg_labels, "$(alg) - $(mf_lab[mf])")
            push!(alg_nll, last(nlls[alg][mf]))
            push!(alg_acc, last(accs[alg][mf]))
            push!(alg_name, string(alg))
            push!(alg_type, string(mf))
            push!(eces, sum(abs2, conf_accs[alg][mf][end] - confs[alg][mf][end]))
        end
    end
end

p1 = bar(alg_labels, alg_nll, ylabel="NLL", label="", lw=0.0)
p2 = bar(alg_labels, 1 .- alg_acc, ylabel="Class. Error", label="", lw=0.0)
savefig(p1, joinpath(plot_dir, savename("nll", @dict(n_hidden, activation, alpha, L, eta), "png")))
savefig(p2, joinpath(plot_dir, savename("err", @dict(n_hidden, activation, alpha, L, eta), "png")))
display(plot(p1, p2))

### Create a dataframe out of the result

d = DataFrame([alg_name, alg_type, alg_nll, alg_acc, eces], [:name, :mf_type, :nll, :acc, :ece])
open(plotsdir("bnn", "tex_results_L=$(L)_" * model * ".tex"), "w") do io
    print(io, latexify(d; env=:table, fmt=x->round(x, sigdigits=3)))
end

## Convergence plots
plt_nll = Dict()
plt_nll[:full] = plot(legend=:topright, title="Mean-Field", yaxis="NLL", xaxis="Iterations")
plt_nll[:none] = plot(legend=:topright, title="L=$L", yaxis="NLL", xaxis="Iterations")
plt_acc = Dict()
plt_acc[:full] = plot(title="Mean-Field", yaxis="Class. Error", xaxis="Iterations")
plt_acc[:none] = plot(title="L=$L", yaxis="Class. Error", xaxis="Iterations")
for alg in bnn_algs[bnn_algs .!= :swag]
    for mf in algs_to_mf[alg]
        if !isnothing(nlls[alg][mf])
            N = length(nlls[alg][mf])
            if mf == :full
                plot!(plt_nll[mf], range(0, n_iter-1, length=N), nlls[alg][mf], color=alg_col[alg], label=alg_lab[alg])
                plot!(plt_acc[mf], range(0, n_iter-1, length=N), 1.0 .- accs[alg][mf], color=alg_col[alg], label=alg_lab[alg])
            else
                plot!(plt_nll[:none], range(0, n_iter-1, length=N), nlls[alg][mf], color=alg_col[alg], label=ece_labs[(alg, mf)])
                plot!(plt_acc[:none], range(0, n_iter-1, length=N), 1.0 .- accs[alg][mf], color=alg_col[alg], label=ece_labs[(alg, mf)])
            end
        end
    end
end
for mf in [:full, :none]
    savefig(plt_nll[mf], joinpath(plot_dir, savename("convergence_nll_$(string(mf))_", @dict(L, activation, alpha, eta, n_hidden), "png")))
    savefig(plt_nll[mf], joinpath(plot_dir, savename("convergence_nll_$(string(mf))_", @dict(L, activation, alpha, eta, n_hidden), "eps")))
    savefig(plt_acc[mf], joinpath(plot_dir, savename("convergence_acc_$(string(mf))_", @dict(L, activation, alpha, eta, n_hidden), "png")))
    savefig(plt_acc[mf], joinpath(plot_dir, savename("convergence_acc_$(string(mf))_", @dict(L, activation, alpha, eta, n_hidden), "eps")))
    display(plot(plt_nll[mf], plt_acc[mf]))
end
# display(plot(plt_nll, plt_acc))
## 
# for (i, n_particles) in enumerate(n_ps)
#     for mf in mfs
#         nll[:gpf][mf][i] = [nlls[:gpf][mf][(α, n_particles)] for α in αs]
#     end
#     nll[:advi][i] = [nlls[:advi][(α, n_particles)] for α in αs]
#     nll[:swag][i] = [nlls[:swag][(α, n_particles)] for α in αs]
#     for mf in mfs
#         acc[:gpf][mf][i] = [accs[:gpf][mf][(α, n_particles)] for α in αs]
#     end
#     acc[:advi][i] = [accs[:advi][(α, n_particles)] for α in αs]
#     acc[:swag][i] = [accs[:swag][(α, n_particles)] for α in αs]
# end
# for (i, n_particles) in enumerate(n_ps)
#     p = plot(xlabel = "α", legend = :topright, xaxis = :log, yaxis = "Neg. Log-Likelihood")#, title = "# particles : $n_particles")
#     for mf in mfs
#         plot!(αs, nll[:gpf][mf][i], label = "GPF - $(mfdict[mf])", linestyle = gpfl[mf], color = gpfc[mf])
#     end
#     plot!(αs, nll[:advi][i], label = "GVA", color = colors[2])
#     plot!(αs, nll[:swag][i], label = "SWAG", color = colors[4], linestyle = :dash)
#     savefig(plotsdir("bnn", savename("nll_lenet_mnist", @dict(n_particles), "png")))
#     display(p)

#     p = plot(xlabel = "α", legend = :bottomright, xaxis = :log, yaxis = "Accuracy")#, title = "# particles : $n_particles")
#     for mf in mfs
#         plot!(αs, acc[:gpf][mf][i], label = "GPF - $(mfdict[mf])", linestyle = gpfl[mf], color = gpfc[mf])
#     end
#     plot!(αs, acc[:advi][i], label = "GVA", color = colors[2])
#     plot!(αs, acc[:swag][i], label = "SWAG", color = colors[4], linestyle = :dash)
#     display(p)
#     savefig(plotsdir("bnn", savename("accuracy_lenet_mnist", @dict(n_particles), "png")))

# end


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
