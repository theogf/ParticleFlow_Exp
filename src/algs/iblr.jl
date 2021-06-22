"""
    IBLR : iBayes Learning Rule, Lin 2020
    x = N(μ, S^{-1})
"""
struct IBLR{T, Tμ<:AbstractVector{T}, TS<:AbstractMatrix{T}} <: VIScheme
    μ::Tμ
    S::TS
    nSamples::Int
end

function IBLR(μ::AbstractVector, S::AbstractMatrix, nSamples::Int=100)
    IBLR(μ, S, nSamples)
end

Distributions.dim(d::IBLR) = length(d.μ)
Distributions.mean(d::IBLR) = d.μ
Distributions.cov(d::IBLR) = inv(d.S)

function Distributions._rand!(
  rng::AbstractRNG,
  d::IBLR,
  x::AbstractVecOrMat,
)
  Distributions._rand!(rng, MvNormalCanon(d.S * d.μ, d.S), x)
end

function update!(d::IBLR, logπ, opt)
    opt isa Descent || error("IBLR only works with std grad. descent")
    t = opt.eta
    θ = rand(d, nSamples(d))

    gμ = -vec(mean(gradcol(d, logπ, θ), dims=2))
    gS = -mean(hessian.(logπ, eachcol(θ)))

    G = d.S - gS
    Δμ = S \ gμ
    d.μ .-= t * Δμ
    d.S .= Symmetric((1 - t) * d.S + t * gS + 0.5 * t^2 * G * (S \  G))
end

function ELBO(d::IBLR, logπ; nSamples::Int=nSamples(d))
    sum(logπ, eachcol(rand(d, nSamples))) / nSamples - 0.5 * logdet(d.S)
end