using Flux
using ProgressMeter
using LinearAlgebra
using Distributions
using StatsBase

struct LowRankGaussianDenseLayer{A,Tμ,Tv,Tσ}
    K::Int # rank
    in::Int
    out::Int
    μ::Tμ # Mean
    v::Tv # Low rank covariance
    σ::Tσ # Diagonal covariance
    a::A # Activation function
    α::Real
end

function LowRankGaussianDenseLayer(in, out, a, K, α=1/K)
    μ = randn((in + 1) * out)
    σ = rand((in + 1) * out)
    v = randn((in + 1) * out, K)
    LowRankGaussianDenseLayer(K, in, out, μ, v, σ, a, α)
end

function LowRankGaussianDenseLayer(l::Dense, K, α=1/K)
    out, in = size(l.W)
    θ = vcat(l.W[:], l.b)
    Σ = Matrix{Float64}()

    LowRankGaussianDenseLayer(K, in, out, θ, ran)
end

Flux.@functor LowRankGaussianDenseLayer

function to_weights_and_bias(l::LowRankGaussianDenseLayer, x::AbstractVector)
    return reshape(x[1:(l.out * l.in)], l.out, l.in), x[(l.out * l.in + 1):end]
end

function (l::LowRankGaussianDenseLayer{A})(x::AbstractArray) where {A}
    B = size(x, 2)
    Wμ, bμ = to_weights_and_bias(l, l.μ)
    Fμ = Dense(Wμ, bμ, identity) # NN from the mean
    Wσ, bσ = to_weights_and_bias(l, l.σ)
    Fσ = Dense(abs2.(Wσ), abs2.(bσ), identity) ## NN from the diagonal covariance
    lr_weights_and_bias = [to_weights_and_bias(l, v) for v in eachcol(l.v)]
    Fvs = [Dense(W, b, identity) for (W, b) in lr_weights_and_bias] # NNs from the Low-Rank
    Yμ = Fμ(x) # Output from the mean
    Yσ = sqrt.(Fσ(x.^2)) # Output from the diagonal covariance
    Yv = sum(Fvs) do Fv
        randn(1, B) .* Fv(x)
    end # Sum of the Low-Rank contributions
    ϵ = randn(l.out, B)
    return l.a.(Yμ + ϵ .* Yσ + sqrt(l.α) * Yv) # Summed output
end

function KLdivergence(l::Chain, γ::Real)
    sum(Base.Fix2(KLdivergence, γ), l)
end

function KLdivergence(l::LowRankGaussianDenseLayer, γ::Real)
    D = (l.in + 1) * l.out
    return 0.5 * (sum(abs2, l.σ) / γ - sum(log ∘ abs2, l.σ)
                    + l.α / γ * sum(abs2, l.v) + sum(abs2, l.μ) / γ -
                    - logdet(I + l.α * l.v' * Diagonal(inv.(l.σ .^ 2)) * l.v)
                    + D * (log(γ) - 1)
                )
end

function pred_mean_and_var(l, x; T=100)
    return vec.(mean_and_var([l(x) for _ in 1:T]))
end

function pred_mean_and_var2(l, x; T=100)
    return vec.(mean_and_var([network_sample(l)(x) for _ in 1:T]))
end

function network_sample(c::Chain)
    Chain(network_sample.(c)...)
end
function network_sample(l::LowRankGaussianDenseLayer)
    θ = rand(MvNormal(l.μ, Symmetric(l.α * l.v * l.v') + Diagonal(l.σ.^2)))
    Dense(to_weights_and_bias(l, θ)..., l.a)
end

## Run a test
N = 50
x = (rand(1, N) * 10) .- 5
y = sin.(x[:])
y = abs.(x[:])
# y = 2 * x[:]
y = y + randn(N) * 0.1
score(y, ŷ) = 0.5 * sum(abs2, y .- ŷ)
neg_elbo(l, x, y) = score(y, l(x)) + KLdivergence(l, 100.0) 
loss(l, x, y) = Flux.Losses.mse(l(x), y)
n_hidden = 10
K = 10
l = Chain(LowRankGaussianDenseLayer(1, n_hidden, relu, K), LowRankGaussianDenseLayer(n_hidden, 1, identity, K))
# l = Chain(Dense(1, 5, relu), Dense(5, 1, identity))
ps = params(l)
opt = ADAM(0.01)
T = 2000
@showprogress for i in 1:T
    if i % 100 == 0
        @info "Current loss: $(neg_elbo(l, x, y')) "
    end
    grads = gradient(ps) do
        neg_elbo(l, x, y')
    end
    Flux.Optimise.update!(opt, ps, grads)
    if i % 1000 == 0
        plot_results() |> display
    end
end
plot_results()

## Plotting test

function plot_results()
    x_test = range(-7, 7, length=100)'
    scatter(x', y, lab="")
    y_test_m, y_test_v = pred_mean_and_var2(l, x_test)
    plot!(x_test', y_test_m, ribbon=sqrt.(y_test_v), lab="")
    plot!(x_test', [vec(network_sample(l)(x_test)) for _ in 1:10], lab="")
end