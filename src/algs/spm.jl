"""
    SPM : Sparse Precision Matrix, Titsias 2014
    θ=μ+L^{-1}z
"""
struct SPM{S, Tμ<:AbstractVector{S}, TT<:AbstractMatrix{S}} <: VIScheme
    μ::Tμ
    T::TT
    invT::TT
    T′::TT
    nSamples::Int
end

function SPM(μ::AbstractVector, T::LowerTriangular, nSamples::Int=100)
  T′ = copy(T)
  setdiag!(T′, log.(diag(T)))
  SPM(μ, T, inv(T), T′, nSamples)
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
    s = randn(dim(d), nSamples(d))
    θ = d.invT * s .+ d.μ
    g = gradcol(logπ, θ)
    
    gμ = vec(mean(g + d.T * s, dims=2))
    Δμ = Optimise.apply!(opt, d.μ, gμ)
    d.μ .+= Δμ

    gT′ = - d.invT * s * g' * d.invT
    muldiag!(gT′, diag(d.T))
    ΔT′ = Optimise.apply!(opt, d.T′, gT′)
    d.T′ .+= LowerTriangular(ΔT′)
    d.T .= d.T′
    setdiag!(d.T, exp.(diag(d.T′)))
    d.invT .= inv(d.T)
end

function ELBO(d::SPM, logπ; nSamples::Int=nSamples(d))
    sum(logπ, eachcol(rand(d, nSamples))) - logdet(d.T)
end