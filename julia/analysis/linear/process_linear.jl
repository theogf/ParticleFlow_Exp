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
B = 200
n_particles = 100
α = 0.01
σ_init = 1

## Get ADVI data
mf = :partial
prefix = datadir("results", "linear", dataset)
nruns = 1
advi_res = [Dict() for i in 1:nruns]
for i in 1:1
    advi_path = joinpath(prefix, savename(@dict i B n_particles α σ_init) * "_advi")
    res = collect_results!(advi_path)
    files = readdir(advi_path)
    iter = parse.(Int64, getindex.(files, range.(12, length.(files).-5; step = 1)))
    res.iter = iter
    last_res = @linq res |> where(:iter .== maximum(:iter))
    advi_acc = Float64[]
    advi_nll = Float64[]
    for q in res.q[sortperm(iter)]
        pred, sig_pred = StatsBase.mean_and_var(x -> logistic.(X_test * x), rand(q, 100))
        acc = mean((pred .> 0.5) .== y_test); push!(advi_acc, acc)
        nll = Flux.Losses.binarycrossentropy(pred, y_test); push!(advi_nll, nll)
    end
    advi_res[i][:iter] = sort(iter)
    advi_res[i][:nll] = advi_nll
    advi_res[i][:acc] = advi_acc
end


## Get GPF data

mf = :none
prefix = datadir("results", "linear", dataset)
nruns = 1
gpf_res = [Dict() for i in 1:nruns]
for i in 1:1
    gpf_path = joinpath(prefix, savename(@dict B n_particles α σ_init) * "_gflow_iter=1")
    res = collect_results!(gpf_path)
    files = readdir(gpf_path)
    iter = parse.(Int64, getindex.(files, range.(12, length.(files).-5; step = 1)))
    res.iter = iter
    last_res = @linq res |> where(:iter .== maximum(:iter))
    gpf_acc = Float64[]
    gpf_nll = Float64[]
    for q in res.q[sortperm(iter)]
        pred, sig_pred = StatsBase.mean_and_var(x -> logistic.(X_test * x), q)
        acc = mean((pred .> 0.5) .== y_test); push!(gpf_acc, acc)
        nll = Flux.Losses.binarycrossentropy(pred, y_test); push!(gpf_nll, nll)
    end
    gpf_res[i][:iter] = sort(iter)
    gpf_res[i][:nll] = gpf_nll
    gpf_res[i][:acc] = gpf_acc
end


## Load Stein data
n_particles = 10
mf = :none
prefix = datadir("results", "linear", dataset)
nruns = 1
stein_res = [Dict() for i in 1:nruns]
for i in 1:1
    stein_path = joinpath(prefix, savename(@dict B i n_particles α σ_init) * "_stein")
    res = collect_results!(stein_path)
    files = readdir(stein_path)
    iter = parse.(Int64, getindex.(files, range.(12, length.(files).-5; step = 1)))
    res.iter = iter
    last_res = @linq res |> where(:iter .== maximum(:iter))
    stein_acc = Float64[]
    stein_nll = Float64[]
    for q in res.q[sortperm(iter)]
        pred, sig_pred = StatsBase.mean_and_var(x -> logistic.(X_test * x), q)
        acc = mean((pred .> 0.5) .== y_test); push!(stein_acc, acc)
        nll = Flux.Losses.binarycrossentropy(pred, y_test); push!(stein_nll, nll)
    end
    stein_res[i][:iter] = sort(iter)
    stein_res[i][:nll] = stein_nll
    stein_res[i][:acc] = stein_acc
end
