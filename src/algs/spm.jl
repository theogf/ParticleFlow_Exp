"""
    SPM : Sparse Precision Matrix, Titsias 2014
    θ=μ+L^{-1}z
"""
struct SPM{T, Tμ<:AbstractVector{T}, TT<:AbstractMatrix{T}} <: VIScheme
    μ::Tμ
    T::TT
    invT::TT
    T′::TT
    nSamples::Int
end

function SPM(μ::AbstractVector, T::AbstractMatrix)
    SPM(μ, T, inv(T), zero(T))
end

Distributions.dim(d::SPM) = length(d.μ)
Distributions.mean(d::SPM) = d.μ
Distributions.cov(d::SPM) = XXt(d.invT)

function Distributions._rand!(
  rng::AbstractRNG,
  d::SPM,
  x::AbstractVector,
)
  nDim = length(x)
  nDim == dim(d) || throw(DimensionMismatch("Wrong dimensions"))
  x .= mean(d) + d.invT * randn(rng, T, dim(d))
end

function Distributions._rand!(
  rng::AbstractRNG,
  d::SPM,
  x::AbstractMatrix,
)
  nDim, nPoints = size(x)
  nDim == dim(d) || throw(DimensionMismatch("Wrong dimensions"))
  x .= mean(d) .+ d.invT * randn(rng, T, dim(d), nPoints)
end

function update!(d::SPM, logπ, opt)
    s = rand(d.φ, nSamples(d))
    θ = d.invT * s + d.μ
    g = gradcol(logπ, θ)
    Δμ = Optimise.apply!(opt, d.μ, mean(g) + mean(d.T * s, dims=2))
    g′ = - mean(d.invT * s * (d.invT * g)', dims = 2)
    muldiag!(g′, diag(d.T))
    ΔT′ = Optimise.apply!(opt, d.T′, g′)
    d.μ .+= Δμ
    d.T′ .+= ΔT′
    d.T = d.T′
    setdiag!(d.T, exp(diag(d.T′)))
    d.invT = inv(d.T)
end

function ELBO(d::SPM, logπ; nSamples::Int=nSamples(d))
    sum(logπ, eachcol(rand(d, nSamples))) - logdet(d.T)
end