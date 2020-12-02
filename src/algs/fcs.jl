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

function FCS(μ::AbstractVector, B::AbstractMatrix, D::AbstractVector=one(μ), nSamples::Int=100)
    FCS(μ, B, Diagonal(D), nSamples)
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
    θ = d.D * ϵ + d.B * z .+ d.μ
    g = gradcol(logπ, θ)
    A = computeA(d.B, d.D)
    Δμ = Optimise.apply!(opt, d.μ, vec(mean(g, dims=2)))
    ΔB = Optimise.apply!(opt, d.B, gradB(g, ϵ, z, d.B, d.D, A))
    ΔD = Optimise.apply!(opt, d.D, gradD(g, ϵ, z, d.B, d.D, A))
    d.μ .+= Δμ
    d.B .+= ΔB
    d.D .+= ΔD
end

function computeA(B, D)
    Dinv = inv(D^2)
    return Dinv - Dinv * B * inv(I + B' * Dinv * B) * B' * Dinv
end

function gradB(g, ϵ, z, B, D, A)
    return (g * z' + A * (B * z + D * ϵ) * z') / size(z, 2)
end

function gradD(g, ϵ, z, B, D, A)
    return Diagonal(vec(mean(g .* ϵ, dims =2)) + diag(A * (B * z + D * ϵ) * ϵ') / size(ϵ, 2))
end

function ELBO(d::FCS, logπ; nSamples::Int=nSamples(d))
    A = computeA(d.B, d.D)
    sum(x->logπ(x) + 0.5 * dot(cov(d), x - mean(d)) , eachcol(rand(d, nSamples))) / nSamples + 0.5 * logdet(cov(d))
end