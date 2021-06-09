using Flux
using ProgressMeter
using LinearAlgebra
using Distributions
using Random
using Functors
using RandomizedLinAlg

struct SLANG{Tμ,TU,Td,T}
    L::Int # Rank
    μ::Tμ # Mean
    U::TU # Low-Rank covariance
    d::Td # Diagonal covariance
    α::T
    β::T
    λ::T
end

function SLANG(L, dim, α=1.0, β=1.0, λ=1.0)
    return SLANG(L, randn(dim), randn(dim, L), rand(dim), α, β, λ)
end 

function Random.rand(rng::AbstractRNG, model::SLANG)
    return fast_sample(rng, model.μ, model.U, model.d)
end

function Random.rand(model::SLANG)
    return rand(Random.GLOBAL_RNG, model)
end


function fast_inverse(g, U, d)
    invD = Diagonal(inv.(d))
    A = inv(I + U' * invD * U)
    y = invD * g - invD * U * A * U' * invD * g
    return y
end

function fast_sample(rng::AbstractRNG, μ, U, d)
    invDsqrt = inv.(sqrt.(d))
    ϵ = randn(rng, length(d))
    V = invDsqrt .* U
    W = V' * V
    A = cholesky(W).L
    B = cholesky(I + W).L
    C = A' \ (B - I) / A
    invK = C + W
    y = invDsqrt .* ϵ - ((V / invK) * transpose(V)) * ϵ
    return μ + y
end

function fast_sample(μ, U, d)
    return fast_sample(Random.GLOBAL_RNG, μ, U, d)
end

function fast_eig(δ, U, β, G, L)
    S = rsvd(Symmetric(δ * U * U' + β * G * G'), L, 2)
    return S.U
end
function orthogonalize(A)
    return Array(qr(A).Q)
end

function fast_eig2(δ, U, β, G, L)
    d = length(U, 1)
    A = Symmetric(δ * U * U' + β * G * G')
    E = rand(Uniform(-1, 1), d, L)
    for _ in 1:4
        E = orthogonalize(A * E)
    end
    old_E = E
    E = A * E
    anorm = maximum(norm(E, dims=1) ./ norm(old_E))
    E = orthogonalize(E)
    vals, V = nystrom(E, anorm)
    s = permsort(vals)
    return V[:, s[1:L]]
end
# Return the diagonal of A * A'
function diag_AAt(A)
    vec(sum(abs2, A; dims=2))
end

function step!(alg::SLANG, to_network, loss)
    θ = fast_sample(alg.μ, alg.U, alg.d)
    δ = 1 - alg.β
    G = transpose(first(Flux.Zygote.jacobian(θ) do x
        nn = to_network(x)
        return loss(nn)
    end))
    V = fast_eig(δ, alg.U, alg.β, G, alg.L)
    Δ = δ * diag_AAt(alg.U) + alg.β * diag_AAt(G) - diag_AAt(V)
    alg.U .= V
    @. alg.d = δ * alg.d + Δ + alg.β * alg.λ # Weird dimensions here
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