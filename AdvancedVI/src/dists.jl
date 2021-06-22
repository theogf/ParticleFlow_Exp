## Series of variation of the MvNormal distribution, different methods need different parametrizations ##
abstract type AbstractPosteriorMvNormal{T} <:
              Distributions.ContinuousMultivariateDistribution end

Base.eltype(::AbstractPosteriorMvNormal{T}) where {T} = T
Base.length(d::AbstractPosteriorMvNormal) = d.dim
Distributions.dim(d::AbstractPosteriorMvNormal) = d.dim
Distributions.mean(d::AbstractPosteriorMvNormal) = d.μ
rank(d::AbstractPosteriorMvNormal) = d.dim

function Distributions._rand!(
  rng::AbstractRNG,
  d::AbstractPosteriorMvNormal{T},
  x::AbstractVector,
) where {T}
    Distributions._rand!(rng, MvNormal(d), x)
end

function Distributions._rand!(
  rng::AbstractRNG,
  d::AbstractPosteriorMvNormal{T},
  x::AbstractMatrix,
) where {T}
    Distributions._rand!(rng, MvNormal(d), x)
end

Distributions.MvNormal(d::AbstractPosteriorMvNormal) = Distributions.MvNormal(mean(d), cov(d))

## Series of LowRank representation of the form Σ = Γ * Γ' ##
abstract type AbstractLowRankMvNormal{T} <:
              AbstractPosteriorMvNormal{T} end

function Distributions._rand!(
  rng::AbstractRNG,
  d::AbstractLowRankMvNormal{T},
  x::AbstractVector,
) where {T}
  nDim = length(x)
  nDim == dim(d) || error("Wrong dimensions")
  x .= d.μ + d.Γ * randn(rng, T, rank(d))
end

function Distributions._rand!(
  rng::AbstractRNG,
  d::AbstractLowRankMvNormal{T},
  x::AbstractMatrix,
) where {T}
  nDim, nPoints = size(x)
  nDim == dim(d) || error("Wrong dimensions")
  x .= d.μ .+ d.Γ * randn(rng, T, rank(d), nPoints)
end

Distributions.var(d::AbstractLowRankMvNormal) = vec(sum(d.Γ .* d.Γ, dims = 2))
Distributions.entropy(d::AbstractLowRankMvNormal) = 0.5 * (log2π + logdet(cov(d) + 1e-5I))
rank(d::AbstractLowRankMvNormal) = size(d.Γ, 2)

## Traditional Cholesky representation where Γ is Lower Triangular

struct CholMvNormal{T, Tμ<:AbstractVector{T}, TΓ<:LowerTriangular{T}} <: AbstractLowRankMvNormal{T}
    dim::Int
    μ::Tμ
    Γ::TΓ
    function CholMvNormal(μ::AbstractVector{T}, Γ::LowerTriangular{T}) where {T}
        length(μ) == size(Γ, 1) || throw(DimensionMismatch("μ and Γ have incompatible sizes")) 
        new{T,typeof(μ),typeof(Γ)}(length(μ), μ, Γ)
    end
    function CholMvNormal(
        dim::Int,
        μ::Tμ,
        Γ::TΓ
    ) where {
        T,
        Tμ<:AbstractVector{T},
        TΓ<:LowerTriangular{T},
    }
        length(μ) == size(Γ, 1) || throw(DimensionMismatch("μ and Γ have incompatible sizes")) 
        new{T,Tμ,TΓ}(dim, μ, Γ)
    end
end

Distributions.cov(d::CholMvNormal) = d.Γ * d.Γ'

## Representation via the precision matrix as in Lin et al. 2020 

struct PrecisionMvNormal{T, Tμ<:AbstractVector{T}, TS<:AbstractMatrix{T}} <: AbstractLowRankMvNormal{T}
    dim::Int
    μ::Tμ
    S::TS
    function PrecisionMvNormal(μ::AbstractVector{T}, S::AbstractMatrix{T}) where {T}
        length(μ) == size(S, 1) || throw(DimensionMismatch("μ and S have incompatible sizes")) 
        new{T,typeof(μ),typeof(S)}(length(μ), μ, S)
    end
    function PrecisionMvNormal(
        dim::Int,
        μ::Tμ,
        S::TS
    ) where {
        T,
        Tμ<:AbstractVector{T},
        TS<:AbstractMatrix{T},
    }
        length(μ) == size(S, 1) || throw(DimensionMismatch("μ and S have incompatible sizes")) 
        new{T,Tμ,TS}(dim, μ, S)
    end
end

function Distributions._rand!(
  rng::AbstractRNG,
  d::PrecisionMvNormal,
  x::AbstractVector,
)
  Distributions._rand!(rng, MvNormalCanon(d.S * d.μ, d.S), x)
end
function Distributions._rand!(
  rng::AbstractRNG,
  d::PrecisionMvNormal,
  x::AbstractMatrix,
)
  Distributions._rand!(rng, MvNormalCanon(d.S * d.μ, d.S), x)
end
Distributions.cov(d::PrecisionMvNormal) = Symmetric(inv(d.S))

struct DiagPrecisionMvNormal{T, Tμ<:AbstractVector{T}, TS<:AbstractVector{T}} <: AbstractLowRankMvNormal{T}
    dim::Int
    μ::Tμ
    S::TS
    function DiagPrecisionMvNormal(μ::AbstractVector{T}, S::AbstractVector{T}) where {T}
        length(μ) == size(S, 1) || throw(DimensionMismatch("μ and S have incompatible sizes")) 
        new{T,typeof(μ),typeof(S)}(length(μ), μ, S)
    end
    function DiagPrecisionMvNormal(
        dim::Int,
        μ::Tμ,
        S::TS
    ) where {
        T,
        Tμ<:AbstractVector{T},
        TS<:AbstractVector{T},
    }
        length(μ) == size(S, 1) || throw(DimensionMismatch("μ and S have incompatible sizes")) 
        new{T,Tμ,TS}(dim, μ, S)
    end
end

function Distributions._rand!(
  rng::AbstractRNG,
  d::DiagPrecisionMvNormal,
  x::AbstractVector,
)
  Distributions._rand!(rng, MvNormalCanon(d.S .* d.μ, d.S), x)
end
function Distributions._rand!(
  rng::AbstractRNG,
  d::DiagPrecisionMvNormal,
  x::AbstractMatrix,
)
  Distributions._rand!(rng, MvNormalCanon(d.S .* d.μ, d.S), x)
end
Distributions.cov(d::DiagPrecisionMvNormal) = Diagonal(inv.(d.S))


struct LowRankMvNormal{
    T,
    Tμ<:AbstractVector{T},
    TΓ<:AbstractMatrix{T},
} <: AbstractLowRankMvNormal{T}
    dim::Int
    μ::Tμ
    Γ::TΓ
    function LowRankMvNormal(μ::AbstractVector{T}, Γ::AbstractMatrix{T}) where {T}
        length(μ) == size(Γ, 1) || throw(DimensionMismatch("μ and Γ have incompatible sizes")) 
        new{T,typeof(μ),typeof(Γ)}(length(μ), μ, Γ)
    end
    function LowRankMvNormal(
        dim::Int,
        μ::Tμ,
        Γ::TΓ
    ) where {
        T,
        Tμ<:AbstractVector{T},
        TΓ<:AbstractMatrix{T},
    }
        length(μ) == size(Γ, 1) || throw(DimensionMismatch("μ and Γ have incompatible sizes")) 
        new{T,Tμ,TΓ}(dim, μ, Γ)
    end
end

Distributions.cov(d::LowRankMvNormal) = d.Γ * d.Γ'

@functor LowRankMvNormal

Base.length(d::AbstractLowRankMvNormal) = d.dim

struct BlockMFLowRankMvNormal{
    T,
    Ti<:AbstractVector{<:Int},
    Tμ<:AbstractVector{T},
    TΓ<:AbstractVector{<:AbstractMatrix{T}},
} <: AbstractLowRankMvNormal{T}
    dim::Int
    rank::Int
    id::Ti
    μ::Tμ
    Γ::TΓ
    function BlockMFLowRankMvNormal(
        μ::AbstractVector{T},
        indices::AbstractVector{<:Int},
        Γ::AbstractVector{<:AbstractMatrix{T}}
    ) where {T}
        rank = sum(x -> size(x, 2), Γ)
        return new{T,typeof(indices),typeof(μ),typeof(Γ)}(length(μ), rank, indices, μ, Γ)
    end
    function BlockMFLowRankMvNormal(
        dim::Int,
        rank::Int,
        indices::Ti,
        μ::Tμ,
        Γ::TΓ,
    ) where {T,Ti,Tμ<:AbstractVector{T},TΓ<:AbstractVector{<:AbstractMatrix{T}}}
        return new{T,Ti,Tμ,TΓ}(dim, rank, indices, μ, Γ)
    end
end

function Distributions._rand!(
  rng::AbstractRNG,
  d::BlockMFLowRankMvNormal{T},
  x::AbstractVector,
) where {T}
  nDim = length(x)
  nDim == d.dim || error("Wrong dimensions")
  x .= d.μ + BlockDiagonal(d.Γ) * randn(rng, T, rank(d))
end

function Distributions._rand!(
  rng::AbstractRNG,
  d::BlockMFLowRankMvNormal{T},
  x::AbstractMatrix,
) where {T}
  nDim, nPoints = size(x)
  nDim == d.dim || error("Wrong dimensions")
  x .= d.μ .+ BlockDiagonal(d.Γ) * randn(rng, T, nDim, rank(d))
end

rank(d::BlockMFLowRankMvNormal) = d.rank

Distributions.cov(d::BlockMFLowRankMvNormal) =
    BlockDiagonal(XXt.(d.Γ))

@functor BlockMFLowRankMvNormal

struct MFMvNormal{
    T,
    Tμ<:AbstractVector{T},
    TΓ<:AbstractVector{T},
} <: AbstractPosteriorMvNormal{T}
    dim::Int
    μ::Tμ
    Γ::TΓ
    function MFMvNormal(
        μ::AbstractVector{T},
        Γ::AbstractVector{T}
    ) where {T}
        return new{T,typeof(μ),typeof(Γ)}(length(μ), μ, Γ)
    end
    function MFMvNormal(
        dim::Int,
        μ::Tμ,
        Γ::TΓ
    ) where {T, Tμ<:AbstractVector{T}, TΓ<:AbstractVector{T}}
        return new{T,Tμ,TΓ}(dim, μ, Γ)
    end
end

function Distributions._rand!(
  rng::AbstractRNG,
  d::MFMvNormal{T},
  x::AbstractVector,
  ) where {T}
  nDim = length(x)
  nDim == d.dim || error("Wrong dimensions")
  x .= d.μ + d.Γ .* randn(rng, T, nDim)
end

function Distributions._rand!(
  rng::AbstractRNG,
  d::MFMvNormal{T},
  x::AbstractMatrix,
) where {T}
  nDim, nPoints = size(x)
  nDim == d.dim || error("Wrong dimensions")
  x .= d.μ .+ d.Γ .* randn(rng, T, nDim, nPoints)
end

Distributions.cov(d::MFMvNormal) = Diagonal(abs2.(d.Γ))
@functor MFMvNormal

## Factorized structure from Ong et al. 2017
"""
    FCSMvNormal(μ, Γ, D)
"""
struct FCSMvNormal{T, Tμ<:AbstractVector{T}, TΓ<:AbstractMatrix{T}, TD<:AbstractVector{T}} <: AbstractPosteriorMvNormal{T}
    dim::Int
    μ::Tμ
    Γ::TΓ
    D::TD
    function FCSMvNormal(μ::AbstractVector{T}, Γ::AbstractMatrix{T}, D::AbstractVector{T}) where {T}
        length(μ) == length(D) || error("Different dimensions between μ and D")
        new{T,typeof(μ),typeof(Γ),typeof(D)}(length(μ), μ, Γ, D)
    end
end

function Distributions._rand!(
  rng::AbstractRNG,
  d::FCSMvNormal{T},
  x::AbstractVector,
  ) where {T}
  nDim = length(x)
  nDim == dim(d) || error("Wrong dimensions")
  x .= d.μ + d.Γ * randn(rng, T, size(d.Γ, 2)) + d.D .* randn(rng, T, nDim)
end

function Distributions._rand!(
  rng::AbstractRNG,
  d::FCSMvNormal{T},
  x::AbstractMatrix,
) where {T}
  nDim, nPoints = size(x)
  nDim == d.dim || error("Wrong dimensions")
  x .= d.μ .+ d.Γ * randn(rng, T, size(d.Γ, 2), nPoints) + d.D .* randn(rng, T, nDim, nPoints)
end

Distributions.cov(d::FCSMvNormal) = d.Γ * d.Γ' + Diagonal(abs2.(d.D))

## Particle based distributions ##
abstract type AbstractSamplesMvNormal{T} <:
              AbstractPosteriorMvNormal{T} end

nParticles(d::AbstractSamplesMvNormal) = d.n_particles
Distributions.mean(d::AbstractSamplesMvNormal) = d.μ
Distributions.var(d::AbstractSamplesMvNormal) = var(d.x, dims = 2) .+ 1e-8
Distributions.entropy(d::AbstractSamplesMvNormal) = 0.5 * (log2π + logdet(cov(d) + 1e-5I))

function update_q!(d::AbstractSamplesMvNormal)
    d.μ .= vec(mean(d.x, dims = 2))
    return nothing
end

"""
    SamplesMvNormal(x)

Create a sample based distribution.
"""
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

Distributions.cov(d::SamplesMvNormal) = cov(d.x, dims = 2, corrected=false)

@functor SamplesMvNormal

struct BlockMFSamplesMvNormal{
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
    function BlockMFSamplesMvNormal(
        x::M,
        indices::AbstractVector{<:Int},
    ) where {T,M<:AbstractMatrix{T}}
        K = length(indices) - 1
        μ = vec(mean(x, dims = 2))
        return new{T,M,typeof(indices),typeof(μ)}(size(x)..., K, indices, x, μ)
    end
    function BlockMFSamplesMvNormal(
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

Distributions.cov(d::BlockMFSamplesMvNormal) =
    BlockDiagonal([cov(view(d.x, (d.id[i]+1):d.id[i+1], :), dims = 2) for i = 1:d.K])

@functor BlockMFSamplesMvNormal

struct MFSamplesMvNormal{
    T,
    Tx<:AbstractMatrix{T},
    Tμ<:AbstractVector{T},
} <: AbstractSamplesMvNormal{T}
    dim::Int
    n_particles::Int
    x::Tx
    μ::Tμ
    function MFSamplesMvNormal(
        x::M,
    ) where {T,M<:AbstractMatrix{T}}
        μ = vec(mean(x, dims = 2))
        return new{T,M,typeof(μ)}(size(x)..., x, μ)
    end
    function MFSamplesMvNormal(
        dim::Int,
        n_particles::Int,
        x::Tx,
        μ::Tμ,
    ) where {T,Tx<:AbstractMatrix{T},Tμ<:AbstractVector{T}}
        return new{T,Tx,Tμ}(dim, n_particles, x, μ)
    end
end

Distributions.cov(d::MFSamplesMvNormal) = Diagonal(vec(var(d.x, dims = 2)))
@functor MFSamplesMvNormal

const SampMvNormal = Union{
    MFSamplesMvNormal,
    BlockMFSamplesMvNormal,
    SamplesMvNormal,
    Bijectors.TransformedDistribution{<:AbstractSamplesMvNormal},
}

"""
    EmpiricalDistribution(x::AbstractMatrix)

Distribution entirely defined by its particles. `x` is of dimension `dims` x `n_particles`
"""
struct EmpiricalDistribution{T,M<:AbstractMatrix{T}} <: AbstractPosteriorMvNormal{T}
    dim::Int
    n_particles::Int
    x::M # Dimensions are nDim x nParticles
    function EmpiricalDistribution(x::M) where {T, M<: AbstractMatrix{T}}
        new{T,M}(size(x)..., x)
    end
end

nParticles(d::EmpiricalDistribution) = d.n_particles
function Distributions._rand!(rng::AbstractRNG, d::EmpiricalDistribution, x::AbstractVector)
    nDim = length(x)
    @assert nDim == d.dim "Wrong dimensions"
    x .= d.x[:,rand(rng, 1:d.n_particles)]
end
function Distributions._rand!(rng::AbstractRNG, d::EmpiricalDistribution, x::AbstractMatrix)
    nDim, nPoints = size(x)
    @assert nDim == d.dim "Wrong dimensions"
    x .= d.x[:,rand(rng, 1:d.n_particles, nPoints)]
end
Distributions.mean(d::EmpiricalDistribution) = vec(mean(d.x, dims=2))
Distributions.cov(d::EmpiricalDistribution) = Distributions.cov(d.x, dims = 2)
Distributions.var(d::EmpiricalDistribution) = Distributions.var(d.x, dims = 2)
Distributions.entropy(d::EmpiricalDistribution) = zero(eltype(d)) # Not valid but does not matter for the optimization


## Reparametrization methods for sampling

function reparametrize!(x, q::PrecisionMvNormal, z)
    x .= q.μ .+ cholesky(q.S).L \ z
end

function reparametrize!(x, q::DiagPrecisionMvNormal, z)
    x .= q.μ .+ sqrt.(inv.(q.S)) .* z
end

function reparametrize!(x, q::CholMvNormal, z)
    x .= q.μ .+ q.Γ * z
end

function reparametrize!(x, q::MFMvNormal, z)
    x .= q.μ .+ q.Γ .* z
end

function reparametrize!(x, q::FCSMvNormal, z, ϵ)
    x .= q.μ .+ q.Γ * z + q.D .* ϵ
end