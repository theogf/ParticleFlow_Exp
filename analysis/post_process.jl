using PDMats, LinearAlgebra
using DataFrames, BSON, CSV, ValueHistories
using Distributions, DistributionsAD
using DataFramesMeta
using Plots; pyplot()
default(lw=3.0, legendfontsize = 15.0, labelfontsize = 15.0, tickfontsize = 13.0)
using LaTeXStrings
using ProgressMeter
using ColorSchemes
colors = ColorSchemes.seaborn_colorblind
using Flux
using AdvancedVI; const AVI = AdvancedVI
using StatsFuns
algs = [:gpf, :advi, :steinvi]


#Takes an array of MVHistory and the true distribution and returns the average error and the variance of the error
function process_means(hs, truth, metric = (x, y) -> norm(x - y))
    T = length(first(hs)[:mu])
    Δm = zeros(T)
    varm = deepcopy(Δm)
    vals = getproperty.(getindex.(hs, :mu), :values)
    for i in 1:T
        μs = getindex.(vals, i)
        Δμs = metric.(μs, Ref(truth))
        Δm[i], varm[i] = mean(Δμs), var(Δμs)
    end
    return Δm, varm
end

function process_fullcovs(hs, truth, metric = (x, y) -> norm(x -y))
    T = length(first(hs)[:sig])
    ΔΣ = zeros(T)
    varΣ = deepcopy(ΔΣ)
    vals = getproperty.(getindex.(hs, :sig), :values)
    for i in 1:T
        Σs = getindex.(vals, i)
        ΔΣs = metric.(Σs, Ref(truth))
        ΔΣ[i], varΣ[i] = mean(ΔΣs), var(ΔΣs)
    end
    return ΔΣ, varΣ
end

function process_time(ts::AbstractVector{<:AbstractVector})
    mean(ts), var(ts)
end

function process_time(hs::AbstractVector)
    process_time(extract_time.(hs))
end

## Extract timing in the format [s]
function extract_time(h)

    t_init = get(h, :t_tic)[2]
    t_end = get(h, :t_toc)[2]
    t_diff = cumsum(t_end-t_init)
    t_init[2:end] .-= t_diff[1:end-1]
    return t_init .- get(h, :t_start)[2][1]
end


## Semi-discrete Optimal Transport

function h(x::AbstractVector, v::AbstractVector, y::AbstractVector, ν::AbstractVector, ϵ::Real, c)
    dot(v, ν) - ϵ * logsumexp((v - c.(Ref(x), y)) ./ ϵ .+ log.(ν)) - ϵ
end

function h(x::AbstractVector, v::AbstractVector, y::AbstractVector, ν::AbstractVector, ϵ::Int, c)
    ϵ == 0 || error("ϵ has to be 0")
    dot(v,ν) + mininum(c.(Ref(x), y) .- v)
end

function optim_v(μ::Distribution, y::AbstractVector, ν::AbstractVector, η::Real, N::Int, ϵ::Real, c)
    v = zero(ν); ṽ = zero(ν)
    @showprogress for k in 1:N
        xₖ = rand(μ)
        ṽ .+= η /√(k) * gradient(ν->h(xₖ, ṽ, y, ν, ϵ, c), ν)[1]
        v = ṽ ./ k + (k - 1) / k * v
    end
    return v
end

function optim_v(x::AbstractVector, μ::AbstractVector, y::AbstractVector, ν::AbstractVector, η::Real, N::Int, ϵ::Real, c)
    N_x = length(x)
    v = zero(ν); g = [zero(ν) for i in 1:N_x]; d = zero(ν)
    @showprogress for k in 1:N
        i = rand(1:N_x)
        d .-= g[i]
        g[i] = μ[i] * gradient(ν->h(x[i], v, y, ν, ϵ, c), ν)[1]
        d .+= g[i]
        v .+= η * d
    end
    return v
end

function wasserstein_semidiscrete(μ, y, ν, ϵ; c=(x,y)->norm(x-y), η::Real = 0.1, N::Int = 100, N_MC::Int=2000)
    v = optim_v(μ, y, ν, η, N, ϵ, c)
    return mean(eachcol(rand(μ, N_MC))) do x
        h(x, v, y, ν, ϵ, c)
    end
end

function wasserstein_discrete(x, μ, y, ν, ϵ; c=(x,y)->norm(x-y), η::Real = 0.1, N::Int = 100, N_MC::Int = 200 )
    v = optim_v(x, μ, y, ν, η, N, ϵ, c)
    return mean(rand(x, N_MC)) do x
        h(x, v, y, ν, ϵ, c)
    end
end

function treat_results(::Val{:advi}, res::DataFrame, X_test, y_test; nMC = 100)
    acc = zeros(length(res.i))
    nll = zeros(length(res.i))
    for (i, q) in enumerate(res.q[sortperm(res.i)])
        pred, sig_pred = StatsBase.mean_and_var(x -> logistic.(X_test * x), rand(q, nMC))
        acc[i] = mean((pred .> 0.5) .== y_test)
        nll[i] = Flux.Losses.binarycrossentropy(pred, y_test)
    end
    return acc, nll
end

function treat_results(::Union{Val{:gflow}, Val{:stein}}, res::DataFrame, X_test, y_test; nMC = 100)
    acc = zeros(length(res.i))
    nll = zeros(length(res.i))
    for (i, ps) in enumerate(res.particles[sortperm(res.i)])
        pred, sig_pred = StatsBase.mean_and_var(x -> logistic.(X_test * x), ps)
        acc[i] = mean((pred .> 0.5) .== y_test)
        nll[i] = Flux.Losses.binarycrossentropy(pred, y_test)
    end
    return acc, nll
end
