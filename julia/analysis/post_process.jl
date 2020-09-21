using PDMats, LinearAlgebra
using DataFrames, BSON, CSV, ValueHistories
using Distributions, DistributionsAD
using DataFramesMeta
using Plots; pyplot()
default(lw=3.0, legendfontsize = 15.0, labelfontsize = 15.0, tickfontsize = 13.0)
using LaTeXStrings
using ColorSchemes
colors = ColorSchemes.seaborn_colorblind
using Flux
using AdvancedVI; const AVI = AdvancedVI

algs = [:gpf, :advi, :steinvi]
labels = Dict(:gpf => "GPF", :advi => "GVA", :steinvi => "SVGD")


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
