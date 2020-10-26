using StatsFuns
using DistributionsAD
using Random: AbstractRNG, GLOBAL_RNG

abstract type AbstractSamplesMvNormal{T} <:
              Distributions.ContinuousMultivariateDistribution end

Base.eltype(::AbstractSamplesMvNormal{T}) where {T} = T
function Distributions._rand!(
  rng::AbstractRNG,
  d::AbstractSamplesMvNormal,
  x::AbstractVector,
)
  nDim = length(x)
  nDim == d.dim || error("Wrong dimensions")
  x .= d.x[rand(rng, 1:d.n_particles), :]
end
function Distributions._rand!(
  rng::AbstractRNG,
  d::AbstractSamplesMvNormal,
  x::AbstractMatrix,
)
  nDim, nPoints = size(x)
  nDim == d.dim || error("Wrong dimensions")
  x .= d.x[rand(rng, 1:d.n_particles, nPoints), :]'
end
Distributions.mean(d::AbstractSamplesMvNormal) = d.μ
Distributions.var(d::AbstractSamplesMvNormal) = var(d.x, dims = 2)
Distributions.entropy(d::AbstractSamplesMvNormal) = 0.5 * (log2π + logdet(cov(d) + 1e-5I))

function update_q!(d::AbstractSamplesMvNormal)
    d.μ .= vec(mean(d.x, dims = 2))
    nothing
end

struct SamplesMvNormal{
    T,
    Tx<:AbstractMatrix{T},
    Tμ<:AbstractVector{T},
} <: AbstractSamplesMvNormal{T}
    dim::Int
    n_particles::Int
    x::Tx
    μ::Tμ
    function SamplesMvNormal(x::M) where {T,M<:AbstractMatrix{T}}
        μ = vec(mean(x, dims = 2))
        new{T,M,typeof(μ)}(size(x)..., x, μ)
    end
    function SamplesMvNormal(
        dim::Int,
        n_particles::Int,
        x::Tx,
        μ::Tμ,
    ) where {
        T,
        Tx<:AbstractMatrix{T},
        Tμ<:AbstractVector{T},
    }
        new{T,Tx,Tμ}(dim, n_particles, x, μ)
    end
end

Distributions.cov(d::SamplesMvNormal) = cov(d.x, dims = 2)

@functor SamplesMvNormal

Base.length(d::AbstractSamplesMvNormal) = d.dim
nParticles(d::AbstractSamplesMvNormal) = d.n_particles

struct MFSamplesMvNormal{
    T,
    Tx<:AbstractMatrix{T},
    Ti<:AbstractVector{<:Int},
    Tμ<:AbstractVector{T},
} <: AbstractSamplesMvNormal{T}
    dim::Int
    n_particles::Int
    K::Int
    id::Ti
    x::Tx
    μ::Tμ
    function MFSamplesMvNormal(
        x::M,
        indices::AbstractVector{<:Int},
    ) where {T,M<:AbstractMatrix{T}}
        K = length(indices) - 1
        μ = vec(mean(x, dims = 2))
        return new{T,M,typeof(indices),typeof(μ)}(size(x)..., K, indices, x, μ)
    end
    function MFSamplesMvNormal(
        dim::Int,
        n_particles::Int,
        K::Int,
        indices::Ti,
        x::Tx,
        μ::Tμ,
    ) where {T,Tx<:AbstractMatrix{T},Ti,Tμ<:AbstractVector{T}}
        return new{T,Tx,Ti,Tμ}(dim, n_particles, K, indices, x, μ)
    end
end

Distributions.cov(d::MFSamplesMvNormal) =
    BlockDiagonal([cov(view(d.x, (d.id[i]+1):d.id[i+1], :), dims = 2) for i = 1:d.K])

@functor MFSamplesMvNormal

struct FullMFSamplesMvNormal{
    T,
    Tx<:AbstractMatrix{T},
    Tμ<:AbstractVector{T},
} <: AbstractSamplesMvNormal{T}
    dim::Int
    n_particles::Int
    x::Tx
    μ::Tμ
    function FullMFSamplesMvNormal(
        x::M,
    ) where {T,M<:AbstractMatrix{T}}
        μ = vec(mean(x, dims = 2))
        return new{T,M,typeof(μ)}(size(x)..., x, μ)
    end
    function FullMFSamplesMvNormal(
        dim::Int,
        n_particles::Int,
        x::Tx,
        μ::Tμ,
    ) where {T,Tx<:AbstractMatrix{T},Ti,Tμ<:AbstractVector{T}}
        return new{T,Tx,Tμ}(dim, n_particles, x, μ)
    end
end

Distributions.cov(d::FullMFSamplesMvNormal) = Diagonal(var(d.x, dims = 2))
@functor FullMFSamplesMvNormal

const SampMvNormal = Union{
    FullMFSamplesMvNormal,
    MFSamplesMvNormal,
    SamplesMvNormal,
    TransformedDistribution{<:AbstractSamplesMvNormal},
}

"""
    PFlowVI(n_particles = 100, max_iters = 1000)

Gaussian Particle Flow Inference (PFlowVI) for a given model.
"""
struct PFlowVI{AD} <: VariationalInference{AD}
    max_iters::Int        # maximum number of gradient steps used in optimization
    precondΔ₁::Bool # Precondition the first gradient (mean)
    precondΔ₂::Bool # Precondition the second gradient (cov)
end

# params(alg::SteinVI) = nothing;#params(alg.kernel)

PFlowVI(args...) = PFlowVI{ADBackend()}(args...)
PFlowVI() = PFlowVI(100, true, false)

alg_str(::PFlowVI) = "PFlowVI"

function vi(
    logπ::Function,
    alg::PFlowVI,
    q::AbstractSamplesMvNormal;
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
        logπ,
        [0.0];
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
    alg::PFlowVI,
    q::TransformedDistribution{<:AbstractSamplesMvNormal};
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
        logπ,
        [0.0];
        optimizer = optimizer,
        callback = callback,
        hyperparams = nothing,
        hp_optimizer = nothing,
    )

    # Return updated `Distribution`
    return q
end

function grad!(
    vo,
    alg::PFlowVI{<:ForwardDiffAD},
    q,
    logπ,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    args...,
)
    f(x) = sum(mapslices(z -> phi(logπ, q, z), x, dims = 1))
    chunk_size = getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(q.dist.x), chunk_size))
    config = ForwardDiff.GradientConfig(f, q.dist.x, chunk)
    ForwardDiff.gradient!(out, f, q.dist.x, config)
end

phi(logπ, q, x) = -eval_logπ(logπ, q, x)

function optimize!(
    vo,
    alg::PFlowVI,
    q::SampMvNormal,
    logπ,
    θ::AbstractVector{<:Real};
    optimizer = TruncatedADAGrad(),
    callback = nothing,
    hyperparams = nothing,
    hp_optimizer = nothing,
)
    alg_name = alg_str(alg)
    samples_per_step = nSamples(alg)
    max_iters = alg.max_iters

    optimizer = if optimizer isa AbstractVector #Base.isiterable(typeof(optimizer))
        length(optimizer) == 2 || error("Optimizer should be of size 2 only")
        optimizer
    else
        fill(optimizer, 2)
    end

    diff_result = DiffResults.GradientResult(q.dist.x)

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0)
    else
        0
    end
    Δ₁ = similar(q.dist.μ)
    Δ₂ = similar(q.dist.x)
    time_elapsed = @elapsed while (i < max_iters) # & converged

        _logπ = if !isnothing(hyperparams)
            logπ(hyperparams)
        else
            logπ
        end

        grad!(vo, alg, q, _logπ, θ, diff_result, samples_per_step)

        Δ = DiffResults.gradient(diff_result)

        Δ₁ .= if alg.precondΔ₁
            cov(q.dist) * vec(mean(Δ, dims = 2))
        else
            vec(mean(Δ, dims = 2))
        end
        shift_x = q.dist.x .- q.dist.μ
        compute_cov_part!(Δ₂, q.dist, shift_x, Δ, alg)

        # apply update rule
        Δ₁ .= apply!(optimizer[1], q.dist.μ, Δ₁)
        Δ₂ .= apply!(optimizer[2], q.dist.x, Δ₂)
        @. q.dist.x = q.dist.x - Δ₁ - Δ₂
        update_q!(q.dist)

        if !isnothing(hyperparams) && !isnothing(hp_optimizer)
            Δ = hp_grad(vo, alg, q, logπ, hyperparams)
            Δ = apply!(hp_optimizer, hyperparams, Δ)
            hyperparams .+= Δ
        end
        AdvancedVI.DEBUG && @debug "Step $i" Δ
        PROGRESS[] && (ProgressMeter.next!(prog))

        if !isnothing(callback)
            callback(i, q, hyperparams)
        end
        i += 1
    end

    return q
end

function compute_cov_part!(
    Δ₂::AbstractMatrix,
    q::SamplesMvNormal,
    x::AbstractMatrix,
    Δ::AbstractMatrix,
    alg::PFlowVI,
)
    Δ₂ .= x
    if alg.precondΔ₂
        A = Δ * x' / q.n_particles - I
        B = inv(q.Σ) # Approximation hessian
        # B = Δ * Δ' # Gauss-Newton approximation
        cond = tr(A' * A) / (tr(A^2) + tr(A' * B * A * q.Σ))
        lmul!(cond * A, Δ₂)
    else
        if q.n_particles < q.dim
            mul!(Δ₂, Δ, x' * x, Float32(inv(q.n_particles)), -1.0f0)
            # If N >> D it's more efficient to compute ϕ xᵀ first
        else
            mul!(Δ₂, Δ * x', x, Float32(inv(q.n_particles)), -1.0f0)
        end
    end
end

function compute_cov_part!(
    Δ₂::AbstractMatrix,
    q::MFSamplesMvNormal,
    x::AbstractMatrix,
    Δ::AbstractMatrix,
    alg::PFlowVI,
)
    Δ₂ .= x
    for i = 1:q.K
        xview = x[(q.id[i]+1):q.id[i+1], :]
        # Proceed to the operation :
        # (ψ - I) * x == (1/ N (∑ ϕᵢxᵀᵢ) - I) * x == (1/N ϕ xᵀ - I) * x
        # It is done via mul!(C, A, B, α, β) : C = αAB + βC
        # If D << N it's more efficient to compute xᵀ x first
        if q.n_particles < q.id[i+1] - q.id[i]
            mul!(
                Δ₂[(q.id[i]+1):q.id[i+1], :],
                Δ[(q.id[i]+1):q.id[i+1], :],
                xview' * xview,
                Float32(inv(q.n_particles)),
                -1.0f0,
            )
            # If N >> D it's more efficient to compute ϕ xᵀ first
        else
            mul!(
                Δ₂[(q.id[i]+1):q.id[i+1], :],
                Δ[(q.id[i]+1):q.id[i+1], :] * xview',
                xview,
                Float32(inv(q.n_particles)),
                -1.0f0,
            )
        end
    end
end

function compute_cov_part!(
    Δ₂::AbstractMatrix,
    q::FullMFSamplesMvNormal,
    x::AbstractMatrix,
    Δ::AbstractMatrix,
    alg::PFlowVI,
)
    Δ₂ .= sum((Δ .*= x), dims = 2) .* x
    Δ₂ ./= q.n_particles
end

function (elbo::ELBO)(
    rng::AbstractRNG,
    alg::PFlowVI,
    q::TransformedDistribution{<:SamplesMvNormal},
    logπ::Function,
)

    res = sum(mapslices(x -> -phi(logπ, q, x), q.dist.x, dims = 1))

    if q isa TransformedDistribution
        res += entropy(q.dist)
    else
        res += entropy(q)
    end
    return res
end
