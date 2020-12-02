"""
    FCS : Factorized Covariance Structure, Ong 2017
    θ=μ+Bz+Dϵ
"""
struct FCS{T, Tμ<:AbstractVector{T}, TC<:AbstractMatrix{T}, TD<:Diagonal{T}} <: VIScheme
    μ::Tμ
    B::TC
    D::TD
    nSamples::Int
end

function FCS(μ::AbstractVector, B::AbstractMatrix, D::AbstractVector)
    FCS(μ, B, Diagonal(D))
end

Distributions.dim(d::FCS) = length(d.μ)
Distributions.mean(d::FCS) = d.μ
Distributions.cov(d::FCS) = d.B * d.B' + d.D^2

function Distributions._rand!(
  rng::AbstractRNG,
  d::FCS,
  x::AbstractVector,
)
  nDim = length(x)
  nDim == dim(d) || throw(DimensionMismatch("Wrong dimensions"))
  x .= mean(d) + d.B * randn(rng, T, size(d.B, 2)) + d.D * randn(rng, T, dim(d))
end

function Distributions._rand!(
  rng::AbstractRNG,
  d::FCS,
  x::AbstractMatrix,
)
  nDim, nPoints = size(x)
  nDim == dim(d) || throw(DimensionMismatch("Wrong dimensions"))
  x .= mean(d) + d.B * randn(rng, T, size(d.B, 2), nPoints) + d.D * randn(rng, T, dim(d), nPoints)
end

function update!(d::FCS, logπ, opt)
    ϵ = randn(dim(d), nSamples(d))
    z = randn(size(d.B, 2), nSamples(d))
    θ = d.D * ϵ + d.B * z + d.μ
    g = gradcol(logπ, θ)
    A = computeA(B, D)
    Δμ = Optimise.apply!(opt, d.μ, mean(g))
    ΔB = Optimise.apply!(opt, d.C, gradB(g, ϵ, z, d.B, d.D, A))
    ΔD = Optimise.apply!(opt, d.C, gradD(g, ϵ, z, d.B, d.D, A))
    d.μ .+= Δμ
    d.B .+= ΔB
    d.D .+= ΔD
end

function computeA(B, D)
    Dinv = inv(D^2)
    return Dinv - Dinv * B * inv(I + B' * Dinv * B) * B' * Dinv
end

function gradB(g, ϵ, z, B, D, A)
    return mean(g * z' + A * (B * z + D * ϵ) * z')
end

function gradD(g, ϵ, z, B, D, A)
    return Diagonal(mean(g .* ϵ + A * (B * z + D * ϵ) * ϵ', dims = 2))
end

function ELBO(d::FCS, logπ; nSamples::Int=nSamples(d))
    A = computeA(d.B, d.D)
    sum(x->logπ(x) + 0.5 * dot(cov(d), x - mean(d)) , eachcol(rand(d, nSamples))) / nSamples + 0.5 * logdet(cov(d))
end