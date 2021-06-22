using Flux
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

## Base constructor
function LowRankGaussianDenseLayer{T}(in, out, a, K, α=Float32(1/K)) where {T}
    μ = randn(T, (in + 1) * out)
    σ = rand(T, (in + 1) * out)
    v = randn(T, (in + 1) * out, K)
    LowRankGaussianDenseLayer(K, in, out, μ, v, σ, a, α)
end

## Convert a dense layer to a ELRGVI layer (does not use existing parameters)
function LowRankGaussianDenseLayer(l::Dense, K, α=Float32(1/K))
    T = eltype(l.W)
    LowRankGaussianDenseLayer{T}(reverse(size(l.W))..., l.σ, K, α)
end

Flux.@functor LowRankGaussianDenseLayer

## Convert the array to the weight matrix and bias vector
function to_weights_and_bias(l::LowRankGaussianDenseLayer, x::AbstractVector)
    return @views reshape(x[1:(l.out * l.in)], l.out, l.in), x[(l.out * l.in + 1):end]
end

## Apply the layer on some input
function (l::LowRankGaussianDenseLayer{A})(x::AbstractArray) where {A}
    B = size(x, 2)
    Fμ = Dense(to_weights_and_bias(l, l.μ)..., identity) # NN from the mean
    Wσ, bσ = to_weights_and_bias(l, l.σ)
    Fσ² = Dense(abs2.(Wσ), abs2.(bσ), identity) ## NN from the diagonal covariance
    Fvs = [Dense(to_weights_and_bias(l, v)..., identity) for v in eachcol(l.v)] # NNs from the Low-Rank
    Yμ = Fμ(x) # Output from the mean
    Yσ = sqrt.(Fσ²(x.^2)) # Output from the diagonal covariance
    ϵ_v = similar(l.v, 1, B)
    Yv = sum(Fvs) do Fv
        randn!(DEFAULT_RNG, ϵ_v) .* Fv(x)
    end # Sum of the Low-Rank contributions
    ϵ = randn(DEFAULT_RNG, Float32, l.out, B)
    return l.a.(Yμ + ϵ .* Yσ + sqrt(l.α) * Yv) # Summed output
end

## Compute KL divergence on chain of ELRGVI layers
function LRKLdivergence(l::Chain, γ::Real)
    sum(x->LRKLdivergence(x, γ), l)
end

## Compute KL divergence on a ELGRVI layer
function LRKLdivergence(l::LowRankGaussianDenseLayer, γ::Real)
    D = (l.in + 1) * l.out
    v1 = sum(abs2, l.σ) / γ - sum(log ∘ abs2, l.σ)
    v2 = l.α / γ * sum(abs2, l.v) + sum(abs2, l.μ) / γ
    v3 = - logdet(I + l.α * l.v' * (inv.(l.σ .^ 2) .* l.v))
    v4 = D * (log(γ) - 1)
    return 0.5 * (v1 + v2 + v3 + v4)
end

## Compute mean and variance over multiple runs
# function pred_mean_and_var(l, x; T=100)
    # return vec.(mean_and_var([l(x) for _ in 1:T]))
# end

function pred_mean_and_var(l, x; f=identity, T=100, device=cpu)
    return mean_and_var([cpu(f(device(network_sample(l))(x))) for _ in 1:T])
end

## Sample one NN from a sample of each layer
function network_sample(c::Chain)
    return Chain(network_sample.(c)...)
end

function sample_from_nn(c::Chain, n::Int=1)
    return mapreduce(Base.Fix2(sample_from_nn, n), vcat, c)
end

function sample_from_nn(l::LowRankGaussianDenseLayer, n::Int=1)
    return l.μ .+ sqrt(l.α) * l.v * randn(Float32, size(l.v, 2), n) + abs.(l.σ) .* randn(Float32, length(l.σ), n)
end

## Sample one dense layer from a sample of one layer
function network_sample(l::LowRankGaussianDenseLayer)
    # θ = rand(MvNormal(l.μ, Symmetric(l.α * l.v * l.v') + Diagonal(l.σ.^2)))
    θ = l.μ + sqrt(l.α) * l.v * randn(Float32, size(l.v, 2)) + abs.(l.σ) .* randn(Float32, length(l.σ))
    Dense(to_weights_and_bias(l, θ)..., l.a)
end

## Run a test
# To run the test uncomment the following
# using Plots
function test_elrgvi()
    function plot_results()
        x_test = range(-7, 7, length=100)'
        scatter(x', y, lab="")
        y_test_m, y_test_v = pred_mean_and_var2(l, x_test)
        plot!(x_test', y_test_m, ribbon=sqrt.(y_test_v), lab="")
        plot!(x_test', [vec(network_sample(l)(x_test)) for _ in 1:10], lab="")
    end
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
end


