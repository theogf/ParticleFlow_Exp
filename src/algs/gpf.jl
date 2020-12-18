"""
    GPF : Gaussian Particle Flow
    θ = μ + Γz
"""

struct GPF{T, Tμ<:AbstractVector{T}, TX<:AbstractMatrix{T}} <: VIScheme
    μ::Tμ
    X::TX
    nSamples::Int
    Pμ::Bool
end

function GPF(X::AbstractMatrix, precondμ::Bool=false)
    μ = vec(mean(X, dims=2))
    nSamples = size(X, 2)
    GPF(μ, X, nSamples, precondμ)
end

Distributions.dim(d::GPF) = length(d.μ)
Distributions.mean(d::GPF) = d.μ
Distributions.cov(d::GPF) = cov(d.X, dims=2, corrected=false) + 1e-5 * I

function update!(d::GPF, logπ, opt)
    φ = -gradcol(d, logπ, d.X)
    φ̄ = vec(mean(φ, dims=2))
    ΔX = d.X .- mean(d)
    Δ₁ = Optimise.apply!(opt, d.μ, d.Pμ ? cov(d) * φ̄ : φ̄)
    Δ₂ = Optimise.apply!(opt, d.X, compute_cov_part(φ, d, ΔX))
    @. d.X = d.X - Δ₁ - Δ₂
    d.μ .= vec(mean(d.X, dims=2))
    return nothing
end

function compute_cov_part(φ, d::GPF, X)
    return ((φ * X')/nSamples(d) - I) * X
    Δ₂ = copy(X)
    if nSamples(d) < dim(d)
        mul!(Δ₂, φ, X' * X, Float32(inv(nSamples(d))), -1.0f0)
    else
        mul!(Δ₂, φ * X', X, Float32(inv(nSamples(d))), -1.0f0)
    end
    return Δ₂
end

function ELBO(d::GPF, logπ; nSamples::Int=nSamples(d))
    sum(logπ, eachcol(rand(d, nSamples))) / nSamples + 0.5 * logdet(cov(d))
end