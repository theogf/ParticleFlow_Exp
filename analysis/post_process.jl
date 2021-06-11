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
algs = [
    :gpf,
    :gf,
    :dsvi,
    :fcs,
    :iblr,
    :svgd_linear,
    :svgd_rbf,
]

alg_lab = Dict(
    :gpf => "GPF",
    :gf => "GF",
    :dsvi => "DSVI",
    :fcs => "FCS",
    :iblr => "IBLR",
    :svgd_linear => "Linear SVGD",
    :svgd_rbf => "RBF SVGD"
)

mf_lab = Dict(
    :full => "MF",
    :partial => "Partial MF",
    :none => "",
)

alg_col = Dict(
    :gpf => colors[1],
    :gf => colors[2],
    :dsvi => colors[3],
    :fcs => colors[4],
    :iblr => colors[8],
    :svgd_linear => colors[5],
    :svgd_rbf => colors[6],
)

alg_ls = Dict(
    :gpf => :dash,
    :gf => :dash,
    :dsvi => :solid,
    :iblr => :solid,
    :fcs => :solid,
    :svgd_linear => :solid,
    :svgd_rbf => :solid,
)



alg_line = Dict(
    false => :solid,
    true => :dash,
)

alg_mf_line = Dict(
    :none => :solid,
    :full => :solid,
    :true => :solid,
    :partial => :dash,
)

alg_line_order = [:iblr, :fcs, :svgd_linear, :svgd_rbf, :dsvi, :gf, :gpf]
alg_line_order_dict = Dict(x=>i for (i,x) in enumerate(alg_line_order))
alg_lw = Dict(
    :gpf => 3.0,
    :gf => 3.5,
    :dsvi => 4.0,
    :iblr => 3.0,
    :fcs => 4.5,
    :svgd_linear => 3.0,
    :svgd_rbf => 3.0,
)


#Takes an array of MVHistory and the true distribution and returns the average error and the variance of the error
function process_means(hs, truth; metric = (x, y) -> norm(x - y), use_quantile=false)
    vals = getproperty.(getindex.(hs, :mu), :values)
    global debug_vals = vals
    T = floor(Int, median(length.(vals)))
    Δm = zeros(T)
    varm = use_quantile ? zeros(T, 2) : zeros(T)
    incomplete_runs = findall(x->length(x)!=T, vals)
    deleteat!(vals, incomplete_runs)
    if first(vals) isa AbstractVector{<:AbstractVector}
        nan_runs = findall(x->any(y->any(isnan, y), x), vals)
        deleteat!(vals, nan_runs)
    else
        nan_runs = findall(x->any(isnan, x), vals)
        deleteat!(vals, nan_runs)
    end
    for i in 1:T
        μs = getindex.(vals, i)
        Δμs = metric.(μs, Ref(truth))
        if use_quantile
            Δm[i], varm[i, :] = mean(Δμs), quantile(Δμs, [0.341, 0.682])
        else
            Δm[i], varm[i] = mean(Δμs), var(Δμs)
        end
    end
    return Δm, varm
end

function process_fullcovs(hs, truth; metric = (x, y) -> norm(x -y), use_quantile=false)
    vals = getproperty.(getindex.(hs, :sig), :values)
    T = floor(Int, median(length.(vals)))
    # This removes all the incomplete runs which might create bugs in the estimation
    incomplete_runs = findall(x->length(x) != T, vals)
    deleteat!(vals, incomplete_runs)
    if first(vals) isa AbstractVector{<:AbstractVector}
        nan_runs = findall(x->any(y->any(isnan, y), x), vals)
        deleteat!(vals, nan_runs)
    else
        nan_runs = findall(x->any(isnan, x), vals)
        deleteat!(vals, nan_runs)
    end
    # Compute the difference 
    ΔΣ = zeros(T)
    varΣ = use_quantile ? zeros(T, 2) : zeros(T)
    for i in 1:T
        Σs = getindex.(vals, i)
        ΔΣs = metric.(Σs, Ref(truth))
        if use_quantile
            ΔΣ[i], varΣ[i, :] = mean(ΔΣs), quantile(ΔΣs, [0.341, 0.682])
        else
            ΔΣ[i], varΣ[i] = mean(ΔΣs), var(ΔΣs)
        end
    end
    return ΔΣ, varΣ
end

function process_means_plus_covs(hs, truth, metric = (x, y) -> norm(x - y))
    m = mean(truth)
    C = cov(truth)
    T = length(first(hs)[:x])
    Δ = zeros(T)
    varΔ = deepcopy(Δ)
    qs = SamplesMvNormal.(first.(getproperty.(getindex.(hs, :x), :values)))
    ms = mean.(qs)
    Cs = cov.(qs)
    Δms = metric.(ms, Ref(m))
    ΔCs = metric.(Cs, Ref(C))
    ΔtrCs = abs.(tr.(Cs .- Ref(C)))
    Δs = Δms + ΔCs
    return mean(Δms), var(Δms), mean(ΔCs), var(ΔCs), mean(Δs), var(Δs), mean(ΔtrCs), var(ΔtrCs)
end

function process_time(ts::AbstractVector{<:AbstractVector})
    t_norm = median(length.(ts))
    incomplete_runs = findall(x->length(x)!=t_norm, ts)
    deleteat!(ts, incomplete_runs)
    mean(ts), var(ts)
end

function process_time(hs::AbstractVector, ::Any)
    process_time(extract_time.(hs[2:end]))
end

function process_time(hs::AbstractVector, ::Val{:gpf})
    process_time(extract_time.(hs[2:end]))
end


function get_mean_and_var(hs::AbstractVector, s::Symbol; use_quantile = false)
    val = [get(h, s)[2] for h in hs]
    T = Int(median(length.(val)))
    incomplete_runs = findall(x->length(x)!=T, val)
    deleteat!(val, incomplete_runs)
    m = mean(val)
    v = if use_quantile
        if s == :nll_train || s == :nll_test
            -reduce(hcat, quantile(getindex.(val, i), [0.341, 0.682]) for i in 1:T)
        else
            reduce(hcat, quantile(getindex.(val, i), [0.341, 0.682]) for i in 1:T)
        end
    else
        var(val)
    end
    if s == :nll_train || s == :nll_test
        return -m, v
    else
        return m, v
    end
end


## Extract timing in the format [s]
function extract_time(h)
    t_init = get(h, :t_tic)[2]
    t_end = get(h, :t_toc)[2]
    if length(t_init)!= length(t_end)
        t_min = min(length(t_init), length(t_end))
        t_init = t_init[1:t_min]
        t_end = t_end[1:t_min]
    end
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

function treat_results(::Union{Val{:gf},Val{:fcs},Val{:dsvi},Val{:iblr}}, res::DataFrame, X_test, y_test; nMC = 100)
    acc = zeros(length(res.i))
    nll = zeros(length(res.i))
    for (i, q) in enumerate(res.q[sortperm(res.i)])
        pred, sig_pred = StatsBase.mean_and_var(x -> logistic.(X_test * x), rand(q, nMC))
        acc[i] = mean((pred .> 0.5) .== y_test)
        nll[i] = Flux.Losses.binarycrossentropy(pred, y_test)
    end
    return acc, nll
end

function treat_results(::Val{:gpf}, res::DataFrame, X_test, y_test; nMC = 100)
    acc = zeros(length(res.i))
    nll = zeros(length(res.i))
    for (i, ps) in enumerate(res.particles[sortperm(res.i)])
        pred, sig_pred = StatsBase.mean_and_var(x -> logistic.(X_test * x), ps)
        acc[i] = mean((pred .> 0.5) .== y_test)
        nll[i] = Flux.Losses.binarycrossentropy(pred, y_test)
    end
    return acc, nll
end
