struct GVAR{T, Tμ<:AbstractVector{T}, TΣ<:AbstractMatrix{T}, TK<:AbstractMatrix{T}} <: VIScheme
    μ::Tμ
    Σ::TΣ
    K::TK
end

function GVAR(μ, Σ)
    GVAR(μ, Σ, zero(Σ))
end

Distributions.dim(d::GVAR) = length(d.μ)
Distributions.mean(d::GVAR) = d.μ
Distributions.cov(d::GVAR) = d.Σ

function update!(d::GVAR, logπ, opt)
    g = ForwardDiff.gradient(logπ, θ)
    Δμ = Optimise.apply!(opt, d.μ, g)
    ΔC = Optimise.apply!(opt, d.C, updateC(g, z, d.C))

    d.Σ = inv(inv(d.K) + Λ) 
    d.μ .+= Δμ
    d.C .+= ΔC
end

function ELBO(d::GVAR, logπ; nSamples::Int=100)
    sum(logπ, eachcol(rand(d, nSamples))) + logdet(d.C)
end