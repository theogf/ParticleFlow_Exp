"""
    DSVI : Doubly Stochastic Variational Inference, Titsias 2014
    θ=μ+Cz
"""
struct DSVI{T, Tμ<:AbstractVector{T}, TC<:AbstractMatrix{T}, Tφ<:MultivariateDistribution} <: VIScheme
    μ::Tμ
    C::TC
    φ::Tφ
    nSamples::Int
end

function DSVI(μ::AbstractVector, C::AbstractMatrix, nSamples::Int=100)
    (C isa LowerTriangular || C isa Diagonal) || error("C should be a Diagonal or LowerTriangular matrix")
    DSVI(μ, C, MvNormal(zeros(length(μ)), ones(length(μ))), nSamples)
end

Distributions.dim(d::DSVI) = length(d.μ)
Distributions.mean(d::DSVI) = d.μ
Distributions.cov(d::DSVI) = d.C * d.C'

function Distributions._rand!(
  rng::AbstractRNG,
  d::DSVI,
  x::AbstractVector,
)
  nDim = length(x)
  nDim == dim(d) || throw(DimensionMismatch("Wrong dimensions"))
  x .= mean(d) + d.C * randn(rng, T, dim(d))
end

function Distributions._rand!(
  rng::AbstractRNG,
  d::DSVI,
  x::AbstractMatrix,
)
  nDim, nPoints = size(x)
  nDim == dim(d) || throw(DimensionMismatch("Wrong dimensions"))
  x .= mean(d) .+ d.C * randn(rng, T, dim(d), nPoints)
end

function update!(d::DSVI, logπ, opt)
    z = rand(d.φ, nSamples(d))
    θ = d.C * z .+ d.μ
    g = gradcol(logπ, θ)
    Δμ = Optimise.apply!(opt, d.μ, vec(mean(g, dims=2)))
    ΔC = Optimise.apply!(opt, d.C, updateC(g, z, d.C))
    d.μ .+= Δμ
    d.C .+= ΔC
end

function updateC(g, z, C::Diagonal)
    Diagonal(mean(g .* z, dims=2)) + inv(C)
end

function updateC(g, z, C::LowerTriangular)
    LowerTriangular(g * z' / size(z, 2) + inv(Diagonal(C)))
end

function ELBO(d::DSVI, logπ; nSamples::Int=nSamples(d))
    sum(logπ, eachcol(rand(d, nSamples))) + logdet(d.C)
end