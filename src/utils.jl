invquad(A::AbstractMatrix, x::AbstractVecOrMat) = dot(x, A \ x)
XXt(X::AbstractVecOrMat) = X * X'
gradcol(f, X) = gradient(x->sum(f.(eachcol(x))), X)

function muldiag!(A, v)
    for i in 1:size(A, 1)
        A[i, i] *= v[i] 
    end
end

function setdiag!(A, v)
    for i in 1:size(A, 1)
        A[i, i] = v[i] 
    end
end

function nat_to_meanvar(η₁, η₂)
    invΣ = cholesky(- 2 * η₂)
    μ = invΣ \ η₁
    return μ, inv(invΣ.L)
end

function meanvar_to_nat(μ, L)
    invΣ = XXt(inv(L))
    invΣ * μ, -0.5 * invΣ
end

nat_to_expec(η₁, η₂) = meanvar_to_expec(nat_to_meanvar(η₁, η₂)...)

expec_to_nat(μ, E) = mean_var_to_nat(expec_to_meanvar(μ, E)...)

function expec_to_meanvar(μ, E)
    μ, cholesky(E - XXt(μ)).L
end

function meanvar_to_expec(μ, L)
    μ, XXt(L) + XXt(μ)
end



struct IncreasingRate
    α::Float64 # Maximum learning rate
    γ::Float64 # Convergence rate to the maximum
    state
end

IncreasingRate(α=1.0, γ=1e-8) = IncreasingRate(α, γ, IdDict())

function Optimise.apply!(opt::IncreasingRate, x, g)
    t = get!(()->0, opt.state, x)
    opt.state[x] += 1
    return g .* opt.α * (1 - exp(-opt.γ * t))
end

struct LogLinearIncreasingRate
    γmax::Float64 # Maximum learning rate
    γmin::Float64 # Convergence rate to the maximum
    K::Int
    state
end

LogLinearIncreasingRate(γmax=1.0, γmin=1e-6, K=100) = LogLinearIncreasingRate(γmax, γmin, K, IdDict())

function Optimise.apply!(opt::LogLinearIncreasingRate, x, g)
    t = get!(()->1, opt.state, x)
    γ = 10^(((opt.K - min(t, opt.K)) * log10(opt.γmin) + min(t, opt.K) * log10(opt.γmax))/opt.K)
    opt.state[x] += 1
    return g .* γ
end