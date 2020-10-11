using DrWatson
@quickactivate
include(projectdir("analysis", "post_process.jl"))
include(srcdir("utils", "linear.jl"))
include(srcdir("utils", "tools.jl"))
using AdvancedVI; const AVI = AdvancedVI
using BlockDiagonals
using LinearAlgebra
using DistributionsAD
## Load data
dataset = "bioresponse"
(X_train, y_train), (X_test, y_test) = load_logistic_data(dataset)

## Parameters used
ps = Dict(
    :B => 200,
    :n_particles => 10,
    :α => 0.1,
    :σ_init => 1,
    :cond1 => false,
    :cond2 => false,
    :n_iters => 2001,
    :use_gpu => false,
    :n_runs => 10,
    :seed => 42,
    :dataset => dataset,
    :advi => true,
    :gpf => true,
    :steinvi => true,
    )

# do_run
# n_particles = 2:2:10
# for n_p in n_particles
## Get partial MF
ps[:advi] = true
ps[:steinvi] = false
mf = :partial
prefix_folder = datadir("results", "linear", dataset, savename(merge(ps, @dict mf)))
@assert isdir(prefix_folder) "$prefix_folder"
partialmf = Dict()
partialmodels = [:gflow, :advi]
for model in partialmodels
    partialmf[model] = [Dict() for i in 1:ps[:n_runs]]
    for i in 1:ps[:n_runs]
        model_path = joinpath(prefix_folder, savename(string(model), @dict i))
        res = collect_results!(model_path)
        # last_res = @linq res |> where(:i .== maximum(:i))
        acc, nll = treat_results(Val(model), res, X_test, y_test)
        partialmf[model][i][:acc] = acc
        partialmf[model][i][:nll] = nll
        partialmf[model][i][:iter] = sort(res.i)
    end
end
for model in partialmodels
    res = partialmf[model]
    partialmf[model] = Dict()
    for metric in [:acc, :nll]
        partialmf[model][Symbol(metric, "_m")] = mean(x[metric] for x in res)
        partialmf[model][Symbol(metric, "_v")] = vec(StatsBase.var(reduce(hcat,x[metric] for x in res), dims = 2))
    end
    partialmf[model][:iter] = first(res)[:iter]
end
## Get no-Mean field data
ps[:advi] = false
ps[:steinvi] = true
mf = :none
prefix_folder = datadir("results", "linear", dataset, savename(merge(ps, @dict mf)))
@assert isdir(prefix_folder)
@assert isdir(prefix_folder)
nonemf = Dict()
nonemodels = [:gflow, :stein]
for model in nonemodels
    nonemf[model] = [Dict() for i in 1:ps[:n_runs]]
    for i in 1:ps[:n_runs]
        model_path = joinpath(prefix_folder, savename(string(model), @dict i))
        res = collect_results!(model_path)
        # last_res = @linq res |> where(:i .== maximum(:i))
        acc, nll = treat_results(Val(model), res, X_test, y_test)
        nonemf[model][i][:acc] = acc
        nonemf[model][i][:nll] = nll
        nonemf[model][i][:iter] = sort(res.i)
    end
end
for model in nonemodels
    res = nonemf[model]
    nonemf[model] = Dict()
    for metric in [:acc, :nll]
        nonemf[model][Symbol(metric, "_m")] = mean(x[metric] for x in res)
        nonemf[model][Symbol(metric, "_v")] = vec(StatsBase.var(reduce(hcat,x[metric] for x in res), dims = 2))
    end
    nonemf[model][:iter] = first(res)[:iter]
end
## Load full-mean field data
ps[:advi] = true
ps[:steinvi] = false
mf = :full
prefix_folder = datadir("results", "linear", dataset, savename(merge(ps, @dict mf)))
fullmf = Dict()
fullmodels = [:gflow, :advi]
for model in fullmodels
    fullmf[model] = [Dict() for i in 1:ps[:n_runs]]
    for i in 1:ps[:n_runs]
        model_path = joinpath(prefix_folder, savename(string(model), @dict i))
        if isdir(model_path)
            res = collect_results!(model_path)
            # last_res = @linq res |> where(:i .== maximum(:i))
            acc, nll = treat_results(Val(model), res, X_test, y_test)
            fullmf[model][i][:acc] = acc
            fullmf[model][i][:nll] = nll
            fullmf[model][i][:iter] = sort(res.i)
        else
            fullmf[model][i][:acc] = missing
            fullmf[model][i][:nll] = missing
            fullmf[model][i][:iter] = missing
        end
    end
end
for model in fullmodels
    res = fullmf[model]
    fullmf[model] = Dict()
    for metric in [:acc, :nll]
        fullmf[model][Symbol(metric, "_m")] = mean(skipmissing(x[metric] for x in res))
        fullmf[model][Symbol(metric, "_v")] = vec(StatsBase.var(reduce(hcat, skipmissing(x[metric] for x in res)), dims = 2))
    end
    fullmf[model][:iter] = first(res)[:iter]
end

## Plot the results
plots = []
for metric in [:acc, :nll]
    p = plot(xaxis = "Iteration", yaxis = string(metric))
    # Plotting Full-MF
    # for model in fullmodels
    #     plot!(fullmf[model][:iter], fullmf[model][Symbol(metric, "_m")], ribbon=sqrt.(fullmf[model][Symbol(metric, "_v")]), marker = :o, label = string(model, " - Full MF"))
    # end
    # Plotting Partial-MF
    # for model in partialmodels
    #     plot!(partialmf[model][:iter], partialmf[model][Symbol(metric, "_m")], ribbon=sqrt.(partialmf[model][Symbol(metric, "_v")]), marker = :o, label = string(model, " - Partial MF"))
    # end
    # # Plotting No-MF
    for model in nonemodels
        plot!(nonemf[model][:iter], nonemf[model][Symbol(metric, "_m")], ribbon=sqrt.(nonemf[model][Symbol(metric, "_v")]), marker = :o, label = string(model, " - No MF"))
    end
    push!(plots, p)
    display(p)
end
