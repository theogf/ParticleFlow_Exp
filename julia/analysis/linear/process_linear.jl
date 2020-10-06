using DrWatson
@quickactivate
include(projectdir("analysis", "post_process.jl"))
include(srcdir("utils", "linear.jl"))
include(srcdir("utils", "tools.jl"))
using BlockDiagonals
## Load data
dataset = "swarm_flocking"
(X_train, y_train), (X_test, y_test) = load_logistic_data(dataset)

## Parameters used
ps = Dict(
    :B => 200,
    :n_particles => 8,
    :α => 0.1,
    :σ_init => 1,
    :cond1 => false,
    :cond2 => false,
    :n_iters => 101,
    :use_gpu => false,
    :n_runs => 10,
    :seed => 42,
    :dataset => dataset,
    :advi => true,
    :gpf => true,
    :steinvi => true,
    )


## Get partial MF
ps[:advi] = true
ps[:steinvi] = false
mf = :partial
prefix_folder = datadir("results", "linear", dataset, savename(merge(ps, @dict mf)))
@assert isdir(prefix_folder)
partialmf = Dict()
partialmf[:advi] = [Dict() for i in 1:ps[:n_runs]]
partialmf[:gflow] = [Dict() for i in 1:ps[:n_runs]]
for i in 1:ps[:n_runs]
    for model in [:advi, :gflow]
        model_path = joinpath(prefix_folder, savename(string(model), @dict i))
        res = collect_results!(model_path)
        # last_res = @linq res |> where(:i .== maximum(:i))
        acc, nll = treat_results(Val(model), res, X_test, y_test)
        partialmf[model][i][:acc] = acc
        partialmf[model][i][:nll] = nll
        partialmf[model][i][:iter] = sort(res.i)
    end
end
for model in [:advi, :gflow]
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
nonemf[:stein] = [Dict() for i in 1:ps[:n_runs]]
nonemf[:gflow] = [Dict() for i in 1:ps[:n_runs]]
for i in 1:ps[:n_runs]
    for model in [:stein, :gflow]
        model_path = joinpath(prefix_folder, savename(string(model), @dict i))
        res = collect_results!(model_path)
        # last_res = @linq res |> where(:i .== maximum(:i))
        acc, nll = treat_results(Val(model), res, X_test, y_test)
        nonemf[model][i][:acc] = acc
        nonemf[model][i][:nll] = nll
        nonemf[model][i][:iter] = sort(res.i)
    end
end
for model in [:gflow, :stein]
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
fullmf[:advi] = [Dict() for i in 1:ps[:n_runs]]
fullmf[:gflow] = [Dict() for i in 1:ps[:n_runs]]
for i in 1:ps[:n_runs]
    for model in [:advi, :gflow]
        model_path = joinpath(prefix_folder, savename(string(model), @dict i))
        res = collect_results!(model_path)
        # last_res = @linq res |> where(:i .== maximum(:i))
        acc, nll = treat_results(Val(model), res, X_test, y_test)
        fullmf[model][i][:acc] = acc
        fullmf[model][i][:nll] = nll
        fullmf[model][i][:iter] = sort(res.i)
    end
end
for model in [:gflow, :advi]
    res = fullmf[model]
    fullmf[model] = Dict()
    for metric in [:acc, :nll]
        fullmf[model][Symbol(metric, "_m")] = mean(x[metric] for x in res)
        fullmf[model][Symbol(metric, "_v")] = vec(StatsBase.var(reduce(hcat,x[metric] for x in res), dims = 2))
    end
    fullmf[model][:iter] = first(res)[:iter]
end

## Plot the results
ps = []
for metric in [:acc, :nll]
    p = plot(xaxis = "Iteration", yaxis = string(metric))
    # Plotting Full-MF
    for model in [:gflow, :advi]
        plot!(fullmf[model][:iter], fullmf[model][Symbol(metric, "_m")], ribbon=sqrt.(fullmf[model][Symbol(metric, "_v")]), marker = :o, label = string(model, " - FullMF"))
    end
    # Plotting Partial-MF
    for model in [:gflow, :advi]
        plot!(partialmf[model][:iter], partialmf[model][Symbol(metric, "_m")], ribbon=sqrt.(partialmf[model][Symbol(metric, "_v")]), marker = :o, label = string(model, " - PartialMF"))
    end
    # Plotting Partial-MF
    for model in [:gflow, :stein]
        plot!(nonemf[model][:iter], nonemf[model][Symbol(metric, "_m")], ribbon=sqrt.(nonemf[model][Symbol(metric, "_v")]), marker = :o, label = string(model, " - No MF"))
    end
    push!(ps, p)
    display(p)
end
