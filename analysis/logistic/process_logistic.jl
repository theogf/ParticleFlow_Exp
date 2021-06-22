using DrWatson
@quickactivate
include(projectdir("analysis", "post_process.jl"))
# include(srcdir("utils", "linear.jl"))
include(srcdir("utils", "tools.jl"))
using AdvancedVI; const AVI = AdvancedVI
using LinearAlgebra
using MLDataUtils
## Load data
dataset = "ionosphere"
dataset_file = endswith(dataset, ".csv") ? dataset : dataset * ".csv"
data = CSV.read(datadir("exp_raw", "logistic", dataset_file), DataFrame; header=true)
X = Matrix(data[1:end-1])
y = Vector(data[end])

## Parameters used
ps = Dict(
    :B => -1,
    :n_particles => 10,
    :alpha => 0.1,
    :σ_init => 1.0,
    :natmu => false,
    :n_iters => 2001,
    :use_gpu => false,
    :k => 3,
    :seed => 42,
    :dataset => dataset,
    :gpf => true,
    :gf => true,
    :dsvi => true,
    :fcs => true,
    :iblr => false,
    :eta => 0.01,
    :opt_det => Descent,
    :opt_stoch => RMSProp,
    :comp_hess => :hess,
    :mf => :full,
    )


## Get no-Mean field data
p_nomf = deepcopy(ps)
p_nomf[:mf] = :none
prefix_folder = datadir("results", "logistic", dataset, savename(p_nomf))
isdir(prefix_folder) || error("Prefix folder $(prefix_folder) does not exist")
nonemf = Dict()
nonemodels = [
                :gpf,
                :gf,
                :fcs,
                :dsvi,
                #:iblr,
            ]
for model in nonemodels
    nonemf[model] = [Dict() for i in 1:ps[:k]]
    for (i, ((X_train, y_train), (X_test, y_test))) in enumerate(kfolds((X, y), obsdim=1, k=ps[:k]))
        model_path = joinpath(prefix_folder, savename(string(model), @dict i))
        if isdir(model_path)
            res = collect_results(model_path)
            # last_res = @linq res |> where(:i .== maximum(:i))
            acc, nll = treat_results(Val(model), res, X_test, y_test)
            nonemf[model][i][:acc] = acc
            nonemf[model][i][:nll] = nll
            nonemf[model][i][:iter] = sort(res.i)
        else
            @warn "$(model_path) is not existing"
            nonemf[model][i][:acc] = missing
            nonemf[model][i][:nll] = missing
            nonemf[model][i][:iter] = missing
        end
    end
end
for model in nonemodels
    res = nonemf[model]
    nonemf[model] = Dict()
    for metric in [:acc, :nll]
        nonemf[model][Symbol(metric, "_m")] = mean(skipmissing(x[metric] for x in res))
        nonemf[model][Symbol(metric, "_v")] = vec(StatsBase.var(reduce(hcat,skipmissing(x[metric] for x in res)), dims = 2))
    end
    nonemf[model][:iter] = first(res)[:iter]
end
## Load full-mean field data
p_mf = deepcopy(ps)
p_mf[:mf] = :full
prefix_folder = datadir("results", "logistic", dataset, savename(p_mf))
isdir(prefix_folder) || error("Prefix folder $(prefix_folder) does not exist")
fullmf = Dict()
fullmodels = [
                :gpf,
                :gf,
                :dsvi,
                # :iblr,
            ]
for model in fullmodels
    fullmf[model] = [Dict() for i in 1:ps[:k]]
    for (i, ((X_train, y_train), (X_test, y_test))) in enumerate(kfolds((X, y), obsdim=1, k=ps[:k]))
        model_path = joinpath(prefix_folder, savename(string(model), @dict i))
        if isdir(model_path)
            res = collect_results(model_path)
            # last_res = @linq res |> where(:i .== maximum(:i))
            acc, nll = treat_results(Val(model), res, X_test, y_test)
            fullmf[model][i][:acc] = acc
            fullmf[model][i][:nll] = nll
            fullmf[model][i][:iter] = sort(res.i)
        else
            @warn "$(model_path) is not existing"
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
plot_std = true
mf_structs = [fullmf, nonemf]
text_mf = Dict(
    fullmf => " - Full MF",
    nonemf => " - No MF",
    )
for metric in [:acc, :nll]
    p = plot(xaxis = "Iteration", yaxis = string(metric))
    # Plotting Full-MF
    for mfs in mf_structs
        for model in mfs
            if plot_std
                plot!(mfs[model][:iter], mfs[model][Symbol(metric, "_m")], ribbon=sqrt.(mfs[model][Symbol(metric, "_v")]), marker = :o, label = string(model, text_mf[mfs]))
            else
                plot!(mfs[model][:iter], mfs[model][Symbol(metric, "_m")], marker = :o, label = string(model, text_mf[mfs]))
            end
        end
    end
    # Plotting Partial-MF
    # for model in partialmodels
    #     plot!(partialmf[model][:iter], partialmf[model][Symbol(metric, "_m")], ribbon=sqrt.(partialmf[model][Symbol(metric, "_v")]), marker = :o, label = string(model, " - Partial MF"))
    # end
    # Plotting No-MF
    # for model in nonemodels
    #     if plot_std
    #         plot!(nonemf[model][:iter], nonemf[model][Symbol(metric, "_m")], ribbon=sqrt.(nonemf[model][Symbol(metric, "_v")]), marker = :o, label = string(model, " - No MF"))
    #     else
    #         plot!(nonemf[model][:iter], nonemf[model][Symbol(metric, "_m")], marker = :o, label = string(model, " - No MF"))
    #     end
    # end
    push!(plots, p)
    display(p)
end
plot(plots...)
