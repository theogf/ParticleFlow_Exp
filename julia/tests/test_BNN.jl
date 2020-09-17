# Based on https://turing.ml/dev/tutorials/3-bayesnn/
using Turing, Flux, Plots, Random;
using CUDA
using DistributionsAD
using AdvancedVI, Bijectors
const AVI = AdvancedVI
using ReverseDiff
using Zygote
using ValueHistories
AVI.setadbackend(:zygote)
# AVI.setadbackend(:reversediff)

## Create the data

# Number of points to generate.
N = 160
M = round(Int, N / 4)
Random.seed!(1234)
device = cpu

# Generate artificial data.
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5;
xt1s = Array([[x1s[i] - 0.5; x2s[i] - 0.5] for i = 1:M])
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5;
append!(xt1s, Array([[x1s[i] - 4; x2s[i] - 4] for i = 1:M]))

x1s = rand(M) * 4.5; x2s = rand(M) * 4.5;
xt0s = Array([[x1s[i] - 0.5; x2s[i] - 4] for i = 1:M])
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5;
append!(xt0s, Array([[x1s[i] - 4; x2s[i] - 0.5] for i = 1:M]))

# Store all the data for later.
xs = [xt1s; xt0s] |> device
ts = [ones(2*M); zeros(2*M)] |> device

# Plot data points.
function plot_data()
    x1 = map(e -> e[1], xt1s)
    y1 = map(e -> e[2], xt1s)
    x2 = map(e -> e[1], xt0s)
    y2 = map(e -> e[2], xt0s)

    Plots.scatter(x1,y1, color="red", clim = (0,1))
    Plots.scatter!(x2, y2, color="blue", clim = (0,1))
end

plot_data()

## Create Neural Net parameters

# We construct an architecture for our neural network using Flux
nn = Chain(Dense(2, 3, tanh),
            Dense(3, 2, tanh),
            Dense(2, 1, σ))

# We extract a flattened vector of parameters and the reconstruction function
θ, re = Flux.destructure(nn)
n_params = length(θ) # Here we have 20 parameters
nn_sizes = broadcast(x-> length(x.W) + length(x.b), nn)
nn_indices = vcat(0, cumsum(nn_sizes))

# Calling re(θ) will reconstruct the network

function nn_forward(xs, nn_params::AbstractVector)
    nn = re(nn_params)
    return nn(xs)
end;


## Create model

# Create a regularization term and a Gaussain prior variance term.
alpha = 0.09
sig = sqrt(1.0 / alpha)
Flux.@functor TuringDiagMvNormal
prior_θ = TuringDiagMvNormal(zeros(n_params), sig .* ones(n_params)) |> device

# Specify the probabalistic model.
@model bayes_nn(xs, ts) = begin
    # Create the weight and bias vector.
    nn_params ~ prior_θ

    # Calculate predictions for the inputs given the weights
    # and biases in theta.
    preds = nn_forward(xs, nn_params)

    # Observe each prediction.
    for i = 1:length(ts)
        ts[i] ~ Bernoulli(preds[i])
    end
end;


## Perform inference
GC.gc(true)
CUDA.reclaim()
N_particles = 10
n_iters = 5000

m = bayes_nn(hcat(xs...) |> device, ts |> device)
logπ = Turing.Variational.make_logjoint(m)

Xs = hcat(xs...)
function logπ_simple(θ)
    logprior = logpdf(prior_θ, θ)
    pred = vec(nn_forward(Xs, θ))
    loglike = -Flux.Losses.binarycrossentropy(pred, ts; agg = sum)
    # @show logprior
    # @show loglike
    return logprior + loglike
end
σ_init = 1.0
q =  SamplesMvNormal(randn(n_params, N_particles) * σ_init) |> device
# q = MFSamplesMvNormal(randn(n_params, N_particles) * σ_init, nn_indices) |> device
gvi = PFlowVI(n_iters, true, false)

α = 0.1
opt = [ADAGrad(α), ADAGrad(1e-2)] |> device
# opt = ADAGrad(α)


# nn_forward(hcat(xs...), q.μ)
x_range = collect(range(-6,stop=6,length=25))
y_range = collect(range(-6,stop=6,length=25))
pyplot()
h = MVHistory()
predict(q::AVI.AbstractSamplesMvNormal, x) = nn_forward(x, q.μ)[1] .> 0.5
predict(q::Bijectors.TransformedDistribution, x) = predict(q.dist, x)
predict.(Ref(q), xs)
anim = Animation()
function plot_pred(q, i, anim)
    local_q = cpu(q.dist)
    Z = [nn_forward([x, y], local_q.μ)[1] for x=x_range, y=y_range]
    Z_m = [mean(first(nn_forward([x, y], w)) for w in eachcol(local_q.x)) for x=x_range, y=y_range]
    Z_var = [var(first(nn_forward([x, y], w)) for w in eachcol(local_q.x)) for x=x_range, y=y_range]
    p1 = Plots.contourf(x_range, y_range, Z, title="Prediction (i = $i)",clims=(0,1))
    Plots.scatter!(eachrow(Xs)..., zcolor = ts,lab="", )
    p2 = Plots.contourf(x_range, y_range, Z_m, title="MC Mean", clims = (0,1))
    Plots.scatter!(eachrow(Xs)..., zcolor = ts,lab="", )
    p3 = Plots.contourf(x_range, y_range, Z_var, title="MC Var", color = :blues, clims = (0.0,0.2))
    Plots.scatter!(eachrow(Xs)..., zcolor = ts,lab="", )
    p4 = heatmap(cov(local_q); yflip =true, title = "Covariance", lab="")
    p = plot(p1, p2, p3, p4)
    display(p)
    frame(anim)
end
cb = function(i, q, hp)
    if i % 100 == 0
        push!(h, :acc, i, mean(predict.(Ref(q), xs) .== ts))
        plot_pred(q, i, anim)
    end
end


@time AdvancedVI.vi(logπ_simple, gvi, q, optimizer = opt, callback = cb)

gif(anim, joinpath(@__DIR__, "..", "plots", "gifs", q isa MFSamplesMvNormal ? "bnn_mf.gif" : "bnn.gif"), fps = 5)
## Plotting mean and covariance
pyplot()
p1 = Plots.heatmap(q.Σ, yflip = true)
Plots.plot!([0.5, 6.5, 6.5, 0.5, 0.5], [0.5, 0.5, 6.5, 6.5, 0.5],lab="",color=:black, lw= 3.0)
Plots.plot!([6.5, 9.5, 9.5, 6.5, 6.5], [6.5, 6.5, 9.5, 9.5, 6.5],lab="",color=:black, lw= 3.0)
Plots.plot!([0.5, 9.5, 9.5, 0.5, 0.5], [0.5, 0.5, 9.5, 9.5, 0.5],lab="",color=:black, lw= 3.0)
Plots.plot!([9.5, 15.5, 15.5, 9.5, 9.5], [9.5, 9.5, 15.5, 15.5, 9.5],lab="",color=:black, lw= 3.0)
Plots.plot!([9.5, 17.5, 17.5, 9.5, 9.5], [9.5, 9.5, 17.5, 17.5, 9.5],lab="",color=:black, lw= 3.0)
Plots.plot!([15.5, 17.5, 17.5, 15.5, 15.5], [15.5, 15.5, 17.5, 17.5, 15.5],lab="",color=:black, lw= 3.0)
Plots.plot!([17.5, 19.5, 19.5, 17.5, 17.5], [17.5, 17.5, 19.5, 19.5, 17.5],lab="",color=:black, lw= 3.0)
Plots.plot!([19.5, 20.5, 20.5, 19.5, 19.5], [19.5, 19.5, 20.5, 20.5, 19.5],lab="",color=:black, lw= 3.0)
Plots.plot!([17.5, 20.5, 20.5, 17.5, 17.5], [17.5, 17.5, 20.5, 20.5, 17.5],lab="",color=:black, lw= 3.0)
p2 = Plots.heatmap(reshape(q.μ, 1, :), colorbar= false)
layout = @layout [a{0.1h}
                    b{0.9h}]
p = Plots.plot(p2, p1, layout=layout)
p |> display
savefig(joinpath(@__DIR__, "..", "plots", "bnn", "covariance.png"))
## Plotting predictions

Z = [nn_forward([x, y], q.μ)[1] for x=x_range, y=y_range]
p = Plots.contourf(x_range, y_range, Z, title="Prediction",clims=(0,1))
p |> display
Plots.scatter!(eachrow(hcat(xs...))..., zcolor = ts,lab="", )

## Plot accuracy

p = Plots.plot(h[:acc], title = "Accuracy", lab="")
p |> display
savefig(joinpath(@__DIR__, "..", "plots", "bnn", "accuracy.png"))

## Marginals for each parameter

p = plot(histogram.(eachrow(q.x), bins = 20, label="",normalize = true)...)
p |> display
savefig(joinpath(@__DIR__, "..", "plots", "bnn", "hist_weights.png"))
