using DrWatson
@quickactivate
include(projectdir("analysis", "post_process.jl"))
# include(srcdir("utils", "linear.jl"))
include(srcdir("utils", "tools.jl"))
using AdvancedVI; const AVI = AdvancedVI
using LinearAlgebra
using MLDataUtils
save_times = vcat(1:9, 10:5:99, 100:100:999, 1000:1000:9999, 10000:10000:100000)

## Load data
dataset = "krkp"
dataset_file = endswith(dataset, ".csv") ? dataset : dataset * ".csv"
data = CSV.read(datadir("exp_raw", "logistic", dataset_file), DataFrame; header=true)
X = Matrix(data[1:end-1])
n_samples, n_dim = size(X)
y = Vector(data[end])

## Parameters used
ps = Dict(
    :B => 100,
    :p => 100,
    :n_particles => 100,
    :alpha => 0.1,
    :σ_init => 1.0,
    :natmu => false,
    :n_iters => 2001,
    :k => 10,
    :seed => 42,
    :dataset => dataset,
    :eta => 0,
    :opt_det => :Descent,
    :opt_stoch => :Descent,
    :comp_hess => :hess,
    :mf => :none,
    )


prefix = savename(ps)
models = [
    :gpf,
    :gf,
    :dsvi,
    :fcs,
    #:iblr,
]

isdir(datadir("results", "logistic", dataset)) || error("Results folder does not exist")
results = Dict()
for alg in models
    res = Dict()
    res_file = datadir("results", "logistic", dataset, prefix * "_" * string(alg) * ".bson")
    if !isfile(res_file)
        @warn "Prefix folder $(res_file) does not exist, skipping model $alg"
        continue
    end
    vals = BSON.load(res_file)[:vals]
    if !haskey(vals[1], :t_tic)
        continue
    end
    res[:t_m], res[:t_v] = process_time(vals)
    res[:iter] = get(vals[1], :t_tic)[1]
    for metric in [:acc_train, :acc_test, :nll_train, :nll_test]
        res[Symbol(metric, "_m")], res[Symbol(metric, "_v")] = get_mean_and_var(vals, metric)
    end
    results[alg] = res
end

## Plot results
metric = [:nll_train, :nll_test, :acc_train, :acc_test]
text_mf = Dict(
        :none => " - No MF",
        :full => " - Full MF",
)
show_legend = false
use_time = true
show_std = false
plots = Dict()
for m in metric
    plots[m] = plot(
                title=string(m),
                xaxis=:log,
                legend=false,
                xlabel=use_time ? "Time [s]" : "Iterations"
                )
    for alg in keys(results)#models
        res = results[alg]
        if show_std
            plot!(use_time ? res[:t_m] : save_times[1:length(res[:t_m])], res[Symbol(m, "_m")], ribbon=sqrt.(res[Symbol(m, "_v")]), label=string(alg, text_mf[ps[:mf]]))
        else
            plot!(use_time ? res[:t_m] : save_times[1:length(res[:t_m])], res[Symbol(m, "_m")], label=string(alg, text_mf[ps[:mf]]))
        end
    end
end
plots[:legend] = plot(showaxis=false, hidedecorations=true, grid=false, legendfontsize=10.0, title=dataset)
for alg in keys(results)
    plot!([], [], label=string(alg, text_mf[ps[:mf]]))
end
plot(values(plots)...)


