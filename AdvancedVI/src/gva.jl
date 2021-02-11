abstract type GaussianVariationalApproximation{AD} <: VariationalInference{AD} end

const GVA = GaussianVariationalApproximation

nSamples(alg::GVA) = alg.nSamples

function vi(
    logπ::Function,
    alg::GVA,
    q::AbstractPosteriorMvNormal;
    optimizer = TruncatedADAGrad(),
    callback = nothing,
    hyperparams = nothing,
    hp_optimizer = nothing,
)
    DEBUG && @debug "Optimizing $(alg_str(alg))..."
    # Initial parameters for mean-field approx
    # Optimize
    optimize!(
        elbo,
        alg,
        transformed(q, Identity{1}()),
        logπ;
        optimizer = optimizer,
        callback = callback,
        hyperparams = hyperparams,
        hp_optimizer = hp_optimizer,
    )

    # Return updated `Distribution`
    return q
end

function vi(
    logπ::Function,
    alg::GVA,
    q::TransformedDistribution{<:AbstractPosteriorMvNormal};
    optimizer = TruncatedADAGrad(),
    callback = nothing,
    hyperparams = nothing,
    hp_optimizer = nothing,
)
    DEBUG && @debug "Optimizing $(alg_str(alg))..."
    # Initial parameters for mean-field approx
    # Optimize
    optimize!(
        elbo,
        alg,
        q,
        logπ;
        optimizer = optimizer,
        callback = callback,
        hyperparams = nothing,
        hp_optimizer = nothing,
    )

    # Return updated `Distribution`
    return q
end


# General version
function grad!(
    alg::GVA{<:ForwardDiffAD},
    q,
    logπ,
    x,
    out::DiffResults.MutableDiffResult,
    args...,
)
    f(x) = sum(eachcol(x)) do z
        phi(logπ, q, z)
    end
    chunk_size = getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(x), chunk_size))
    config = ForwardDiff.GradientConfig(f, x, chunk)
    ForwardDiff.gradient!(out, f, x, config)
end


# Version for particles
function grad!(
    alg::GVA{<:ForwardDiffAD},
    q,
    logπ,
    out::DiffResults.MutableDiffResult,
    args...,
)
    f(x) = sum(eachcol(x)) do z
        phi(logπ, q, z)
    end
    chunk_size = getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(q.dist.x), chunk_size))
    config = ForwardDiff.GradientConfig(f, q.dist.x, chunk)
    ForwardDiff.gradient!(out, f, q.dist.x, config)
end

function (elbo::ELBO)(
    ::AbstractRNG,
    alg::GVA,
    q::TransformedDistribution{<:AbstractPosteriorMvNormal},
    logπ::Function,
)
    res = sum(mapslices(x -> -phi(logπ, q, x), rand(q.dist, nSamples(alg)), dims = 1))
    res += entropy(q.dist)
    return res
end


include("dsvi.jl")
include("fcs.jl")
include("gaussflow.jl")
include("gausspartflow.jl")
include("iblr.jl")