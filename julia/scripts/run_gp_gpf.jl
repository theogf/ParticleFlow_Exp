using DrWatson
@quickactivate
include(srcdir("gp", "gp_gpf.jl"))

exp_p = Dict(
    :seed => 42,
    :dataset => "banana",
    :n_particles => 10,
    :n_iters => 10,
    :n_runs => 2,
    :cond1 => false,
    :cond2 => false,
    :σ_init => 1.0,
    :opt => ADAGrad(0.01),
)


run_gp_gpf(exp_p)

## Testing stuff
using Plots
(X_train, y_train), (X_test, y_test) = load_gp_data("banana")
p1 = scatter(eachcol(X_train)..., color = Int.(y_train); lab="")
y_trainpm = sign.(y_train .- 0.5)
n_train, n_dim = size(X_train)
ρ = initial_lengthscale(X_train)
k = transform(SqExponentialKernel(), ρ) + 0.1 * WhiteKernel()

K = kernelpdmat(k, X_train; obsdim = 1)
prior = TuringDenseMvNormal(zeros(n_train), K)
function logπ(θ)
    -Flux.Losses.logitbinarycrossentropy(θ, y_train; agg = sum) + logpdf(prior, θ)
end
neglogπ(θ) = -logπ(θ)
function gradneglogπ!(G, θ)
    G .= - y_trainpm .* logistic.(-y_trainpm .* θ) + (K \ θ) # Correct gradient (verified)
end
opts = Optim.Options(show_every = 5)
neglogπ(Vector(y_trainpm))
x_init = randn(n_train)
x_init .= y_trainpm

g = similar(x_init)
gradneglogπ!(g, x_init)

opt_MAP = optimize(
    neglogπ,
    gradneglogπ!,
    randn(n_train),#x_init,
    LBFGS(),
    Optim.Options(show_every = 1, allow_f_increases = false, iterations = 200),
)

x_MAP = Optim.minimizer(opt_MAP)

p2 = scatter(eachcol(X_train)..., zcolor = logistic.(x_MAP), clims = (0, 1))
p3 = scatter(eachcol(X_train)..., zcolor = logistic.(x_init), clims = (0, 1))
plot(p1, p3, p2)
