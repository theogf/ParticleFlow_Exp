"""
    NGD : Natural Gradient Descent, Salimbeni 2018
    θ=μ+Lz
"""
struct NGD{T, Tμ<:AbstractVector{T}, TL<:LowerTriangular{T}} <: VIScheme
    μ::Tμ
    L::TL
    nSamples::Int
end

function NGD(μ::AbstractVector, L::LowerTriangular, nSamples::Int=100)
    NGD(μ, L, nSamples)
end

Distributions.dim(d::NGD) = length(d.μ)
Distributions.mean(d::NGD) = d.μ
Distributions.cov(d::NGD) = d.L * d.L'

function Distributions._rand!(
  rng::AbstractRNG,
  d::NGD,
  x::AbstractVector,
)
  nDim = length(x)
  nDim == dim(d) || throw(DimensionMismatch("Wrong dimensions"))
  x .= mean(d) + d.L * randn(rng, T, dim(d))
end

function Distributions._rand!(
  rng::AbstractRNG,
  d::NGD,
  x::AbstractMatrix,
)
  nDim, nPoints = size(x)
  nDim == dim(d) || throw(DimensionMismatch("Wrong dimensions"))
  x .= mean(d) .+ d.L * randn(rng, T, dim(d), nPoints)
end

function update!(d::NGD, logπ, opt)
    z = randn(dim(d), nSamples(d))
    ξ = d.μ, d.L
    E, to_expec = destructure(meanvar_to_expec(ξ...))
    dL_dexpec = gradient(E) do E
        μ, L = expec_to_meanvar(to_expec(E)...)
        θ = L * z .+ μ
        mean(logπ, eachcol(θ)) + logdet(L)
    end

    η, to_nat = destructure(meanvar_to_nat(ξ...))
    nat_grad = derivative(0) do t
        reduce(vcat, vec.(nat_to_meanvar((to_nat(η) .+ t .* to_expec(dL_dexpec))...)))
    end
    
    # dξ_dη = jacobian(η) do η
    #     vcat(vec.(nat_to_meanvar(to_nat(η)...))...)
    # end
    # nat_grad2 = dξ_dη * dL_dexpec

    ξ, to_meanvar = destructure(ξ)
    Δμ, ΔL = to_meanvar(nat_grad)

    Δμ = Optimise.apply!(opt, d.μ, Δμ)
    ΔL = Optimise.apply!(opt, d.L, ΔL)
    d.μ .+= Δμ
    d.L .+= ΔL
end

function ELBO(d::NGD, logπ; nSamples::Int=nSamples(d))
    sum(logπ, eachcol(rand(d, nSamples))) + logdet(d.C)
end