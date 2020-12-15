"""
    GF : Gaussian Flow
    θ = μ + Γz
"""

struct GF{T, Tμ<:AbstractVector{T}, TΓ<:AbstractMatrix{T}} <: VIScheme
    μ::Tμ
    Γ::TΓ
    nSamples::Int
    Pμ::Bool
end

function GF(μ::AbstractVector, Γ::AbstractMatrix, nSamples::Int=100, precondμ::Bool=false)
    GF(μ, Γ, nSamples, precondμ)
end

Distributions.dim(d::GF) = length(d.μ)
Distributions.mean(d::GF) = d.μ
Distributions.cov(d::GF) = XXt(d.Γ) + 1e-8 * I

function update!(d::GF, logπ, opt)
    z = randn(size(d.Γ, 2), nSamples(d))
    θ = d.Γ * z .+ d.μ
    φ = -gradcol(d, logπ, θ)
    φ̄ = vec(mean(φ, dims=2))
    d.μ .-= Optimise.apply!(opt, d.μ, d.Pμ ? cov(d) * φ̄ : φ̄)
    d.Γ .-= Optimise.apply!(opt, d.Γ, compute_cov_part(φ, d, θ .- mean(d)))
end

function compute_cov_part(φ, d::GF, X)
    return ((φ * X')/nSamples(d) - I) * d.Γ
    Δ₂ = copy(d.Γ)
    if nSamples(d) < dim(d)
        mul!(Δ₂, φ, X' * X, Float32(inv(nSamples(d))), -1.0f0)
    else
        mul!(Δ₂, φ * X', X, Float32(inv(nSamples(d))), -1.0f0)
    end
    return Δ₂
end

function ELBO(d::GF, logπ; nSamples::Int=nSamples(d))
    sum(logπ, eachcol(rand(d, nSamples))) / nSamples + 0.5 * logdet(cov(d))
end