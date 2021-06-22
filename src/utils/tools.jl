using StatsBase
using Flux.Optimise
# Return true for i = 1,2,3,..10,20,30,..,100,200 etc
iseverylog10(iter) = iter % (10^(floor(Int64, log10(iter + 1)))) == 0
include("optimisers.jl")

# Return the mean and the variance over all particles
# function StatsBase.mean_and_var(f, q::AVI.AbstractSamplesMvNormal)
#     return mean_and_var(f, q.x)
# end

# function StatsBase.mean_and_var(f, q::AVI.SteinDistribution)
#     return mean_and_var(f, q.x)
# end

function StatsBase.mean_and_var(f, x::AbstractMatrix)
    vals = f.(eachcol(x))
    return mean(vals), var(vals)
end

# Preload the data if it has not been yet
function preload(dataset::String, folder::String)
    isdir(datadir("exp_raw", folder)) ? nothing : mkpath(datadir("exp_raw", folder)) # Check the path exists and creates it if not
    isfile(datadir("exp_raw", folder, dataset * ".csv")) ? nothing : resolve(dataset, @__FILE__) # Check the dataset have been loaded and download it if not
end

## Easy transformations from mean covaraince to adapted parametrizaiton
function meancov_to_gf(μ, Σ, mf=false)
    S = mf == Inf ? sqrt.(diag(Σ)) : Matrix(cholesky(Σ).L)
    return (copy(μ), S)
end

function meancov_to_dsvi(μ, Σ, mf=false)
    S = mf == Inf ? sqrt.(diag(Σ)) : cholesky(Σ).L
    return (copy(μ), S)
end

function meancov_to_fcs(μ, Σ, mf=false)
    L = cholesky(Σ).L
    return (copy(μ), Matrix(L - Diagonal(L) / sqrt(2)), diag(L) / sqrt(2))
end

function meancov_to_iblr(μ, Σ, mf=false)
    S = mf == Inf ? sqrt.(inv.(diag(Σ))) : Matrix(Symmetric(inv(Σ)))
    return (copy(μ), S)
end