"""
    EKF : Ensemble Kalman Filtering
"""
struct EKF{T, Tμ<:AbstractVector{T}, TL<:AbstractMatrix{T}} <: VIScheme
    μ::Tμ
    L::TC
    nSamples::int
end

function EKF(μ::AbstractVector, L::AbstractMatrix)
    EKF(μ, C, MvNormal(zeros(length(μ)), Diagonal(ones(length(μ)))))
end

Distributions.dim(d::EKF) = length(d.μ)
Distributions.mean(d::EKF) = d.μ
Distributions.cov(d::EKF) = d.L * d.L'

function Distributions._rand!(
  rng::AbstractRNG,
  d::EKF,
  x::AbstractVector,
)
  nDim = length(x)
  nDim == dim(d) || throw(DimensionMismatch("Wrong dimensions"))
  x .= mean(d) + d.L * randn(rng, T, dim(d))
end

function Distributions._rand!(
  rng::AbstractRNG,
  d::EKF,
  x::AbstractMatrix,
)
  nDim, nPoints = size(x)
  nDim == dim(d) || throw(DimensionMismatch("Wrong dimensions"))
  x .= mean(d) .+ d.L * randn(rng, T, dim(d), nPoints)
end

function update!(d::EKF, logπ, opt)
    z = randn(dim(d), nSamples(d))
    E, to_expec = destructure(meanvar_to_expec(d.μ, d.L))
    dL_dexpec = gradient(E) do E
        μ, L = expec_to_meanvar(to_expec(E)...)
        θ = L * z .+ μ
        mean(logπ, eachcol(θ)) + logdet(L)
    end

    η, to_nat = destructure(meanvar_to_nat(d.μ, d.L))
    dξ_dη = jacobian(η) do η
        vec(nat_to_meanvar(to_nat(η)...))
    end

    ξ, to_meanvar = destructure((d.μ, d.L))
    nat_grad = dξ_dη * dL_dexpec
    Δμ, ΔL = to_meanvar(nat_grad)

    Δμ = Optimise.apply!(opt, d.μ, Δμ)
    ΔL = Optimise.apply!(opt, d.L, ΔL)
    d.μ .+= Δμ
    d.L .+= ΔL
end

function ELBO(d::EKF, logπ; nSamples::Int=nSamples(d))
    sum(logπ, eachcol(rand(d, nSamples))) + logdet(d.C)
end