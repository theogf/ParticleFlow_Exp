using Flux
using ProgressMeter
using LinearAlgebra
using Distributions
using Random
using Functors
using RandomizedLinAlg

"""
    LowRankMatrix

Lazy matrix representation of `A = L * L'` with `size(L) == N, D` with N >> D
`A` is never explicitly computed
"""
struct LowRankMatrix{T,A<:AbstractMatrix{T}} <: AbstractMatrix{T}
    L::A
end

Base.:*(A::LowRankMatrix, X::AbstractMatrix) = A.L * (A.L' * X)
Base.:*(X::AbstractMatrix, A::LowRankMatrix) = (X * A.L) * A.L'
Base.size(A::LowRankMatrix) = (size(A.L, 1), size(A.L, 1))

"""
    SumLowRank

Lazy matrix representation of `A = L1 + L2`, L1 and L2 can be anything, `A` is never explicitly computed.
"""
struct SumLowRank{T,A1<:AbstractMatrix{T},A2<:AbstractMatrix{T}} <: AbstractMatrix{T}
    L1::A1
    L2::A2
end

Base.:*(A::SumLowRank, X::AbstractMatrix) = A.L1 * X + A.L2 * X
Base.:*(X::AbstractMatrix, A::SumLowRank) = X * A.L1 + X * A.L2
Base.size(A::SumLowRank) = size(A.L1)


struct SLANG{RNG,Tμ,TU,Td,T}
    rng::RNG
    L::Int # Rank
    μ::Tμ # Mean
    U::TU # Low-Rank covariance
    d::Td # Diagonal covariance
    α::T
    β::T
    λ::T
end

function SLANG(rng, L, dim, α=1f0, β=1f0, λ=1f0)
    T = promote_type(typeof(α), typeof(β), typeof(λ))
    return SLANG(rng, L, randn(dim), randn(dim, L), rand(dim), T(α), T(β), T(λ))
end

function SLANG(L, dim, α=1f0, β=1f0, λ=1f0)
    return SLANG(Random.GLOBAL_RNG, L, dim, α, β, λ)
end 

function Random.rand(rng::AbstractRNG, model::SLANG)
    return fast_sample(rng, model.μ, model.U, model.d)
end

function Random.rand(model::SLANG)
    return rand(Random.GLOBAL_RNG, model)
end


function fast_inverse(g, U, d)
    invD = Diagonal(inv.(d))
    invA = I + U' * invD * U
    y = invD * g - (invD * ((U / invA) * (U' * (invD * g))))
    return y
end

function fast_sample(rng::AbstractRNG, μ, U, d)
    invDsqrt = inv.(sqrt.(d))
    ϵ = randn(rng, length(d))
    V = invDsqrt .* U
    W = V' * V
    A = cholesky(W).L
    B = cholesky(I + W).L
    C = A' \ ((B - I) / A)
    invK = C + W
    y = invDsqrt .* ϵ - ((V / invK) * (transpose(V) * ϵ))
    return μ + y
end

function fast_sample(μ, U, d)
    return fast_sample(Random.GLOBAL_RNG, μ, U, d)
end

function fast_sample(rng::AbstractRNG, alg::SLANG)
    return fast_sample(rng, alg.μ, alg.U, alg.d)
end

function fast_sample(alg::SLANG)
    return fast_sample(Random.GLOBAL_RNG, alg)
end

function fast_eig(δ, U, β, G, L)
    S = rsvd(
            SumLowRank(
                LowRankMatrix(sqrt(δ) * U),
                LowRankMatrix(sqrt(β) * G),
                ),
            L,
            2,
        )
    # S = rsvd(Symmetric(δ * U * U' + β * G * G'), L, 2)
    return S.U
end
function orthogonalize(A)
    return Array(qr(A).Q)
end

# Return the diagonal of A * A'
function diag_AAt(A)
    vec(sum(abs2, A; dims=2))
end

function step!(alg::SLANG, to_network, loss)
    θ = fast_sample(alg.rng, alg.μ, alg.U, alg.d)
    δ = 1 - alg.β
    G = transpose(first(Flux.Zygote.jacobian(θ) do x
            nn = to_network(x)
            return loss(nn)
        end))
    V = fast_eig(δ, alg.U, alg.β, G, alg.L)
    Δ = δ * diag_AAt(alg.U) + alg.β * diag_AAt(G) - diag_AAt(V)
    alg.U .= V
    @. alg.d = δ * alg.d + Δ + alg.β * alg.λ
    ĝ = vec(sum(G, dims=2)) + alg.λ * alg.μ
    Δμ = fast_inverse(ĝ, alg.U, alg.d)
    @. alg.μ -= alg.α * Δμ
    return nothing
end

## Test SLANG y

using StatsBase
# using Plots
function test_slang()
    N = 50
    x = (rand(1, N) * 10) .- 5
    y = sin.(x[:])
    y = abs.(x[:])
    # y = 2 * x[:]
    y = y + randn(N) * 0.1
    n_hidden = 300
    K = 2
    # l = Chain(LowRankGaussianDenseLayer(1, n_hidden, relu, K), LowRankGaussianDenseLayer(n_hidden, 1, identity, K))
    l = Chain(Dense(1, n_hidden, relu), Dense(n_hidden, 1, identity))
    θ, re = Flux.destructure(l)
    loss(nn) = abs2.(nn(x) - y')
    model = SLANG(K, length(θ), 0.1, 0.1)
    T = 2000
    ## Plotting test

    function pred_mean_and_var(model::SLANG, re, x; T=100)
        return vec.(mean_and_var([re(rand(model))(x) for _ in 1:T]))
    end

    function plot_results()
        x_test = range(-7, 7, length=100)'
        scatter(x', y, lab="")
        y_test_m, y_test_v = pred_mean_and_var(model, re, x_test)
        plot!(x_test', y_test_m, ribbon=sqrt.(y_test_v), lab="")
        plot!(x_test', [vec(re(rand(model))(x_test)) for _ in 1:10], lab="")
    end
    @showprogress for i in 1:T
        if i % 100 == 0
            # @info "Current loss: $(loss(l)) "
        end
        step!(model, re, loss)
        if i % 500 == 0
            plot_results() |> display
        end
    end
    plot_results()


    y_test_m, y_test_v = pred_mean_and_var(model, re, x_test)
    plot_results()
end

# @profview [step!(model, re, loss) for _ in 1:100]