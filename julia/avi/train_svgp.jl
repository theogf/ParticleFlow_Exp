include("train_model.jl")

n_samples = 60
n_iters = 100

## Create some toy data
N = 200
B = 30
M = 20
x = range(0, 1, length = N)
Z = range(0, 1, length = M)
θ = log.([1.0, 10.0, 1e-3])
k = exp(θ[1]) * transform(SqExponentialKernel(), exp(θ[2]))
K = kernelmatrix(k, x) + 1e-5I
f = rand(MvNormal(K))
likelihood(f, θ) = Normal(f, sqrt(exp(θ[3])))
y = rand.(likelihood.(f, Ref(θ)))
plot(x, y)

## Create the model
function meta_logπ(θ)
    k = exp(θ[1]) * transform(SqExponentialKernel(), exp(θ[2]))
    Ku = kernelmatrix(k, Z) + 1e-5I
    Kf = kerneldiagmatrix(k, x) .+ 1e-5
    Kfu = kernelmatrix(k, x, Z)
    P = Kfu / Ku
    d = TuringDenseMvNormal(zeros(length(Z)), Ku)
    return function(z)
        S = sample(1:length(x), B, replace=false)
        length(x) / B * sum(logpdf.(Normal.(y[S], exp(θ[3]) .+ Kf[S] - diag(P[S, :] * Kfu[S,:]')), P[S, :] * z)) + logpdf(d, z)
    end
end
k = exp(θ[1]) * transform(SqExponentialKernel(), exp(θ[2]))
Ku = kernelmatrix(k, Z) + 1e-5I
Kf = kerneldiagmatrix(k, x) .+ 1e-5
Kfu = kernelmatrix(k, x, Z)
P = Kfu / Ku
logπ_reduce = meta_logπ(θ)
logπ_reduce(rand(M))
# AVI.setadbackend(:reversediff)
## Start experiment
hp_init = θ .- 1
hp_init = nothing

opt = ADAGrad(0.2)

general_p =
    Dict(:hyper_params => hp_init, :hp_optimizer => ADAGrad(0.1), :n_dim => M)
gflow_p = Dict(
    :run => true,
    :n_particles => 60,
    :max_iters => n_iters,
    :cond1 => true,
    :cond2 => true,
    :opt => deepcopy(opt),
    :callback => wrap_cb,
    :init => nothing,
)
advi_p = Dict(
    :run => true,
    :n_samples => n_samples,
    :max_iters => n_iters,
    :opt => deepcopy(opt),
    :callback => wrap_cb,
    :init => nothing,
)
stein_p = Dict(
    :run => true,
    :n_particles => n_samples,
    :max_iters => n_iters,
    :kernel => transform(SqExponentialKernel(), 1.0),
    :opt => deepcopy(opt),
    :callback => wrap_cb,
    :init => nothing,
)


g_h, a_h, s_h =
    train_model(x, y, logπ_reduce, general_p, gflow_p, advi_p, stein_p)
    # train_model(x, y, meta_logπ, general_p, gflow_p, advi_p, stein_p)

## Plotting

iters, mus_g = get(g_h, :mu)
iters, mus_a = get(a_h, :mu)
iters, mus_s = get(s_h, :mu)

g = @gif for (i, mu_g, mu_a, mu_s) in zip(iters, mus_g, mus_a, mus_s)
    plot(x, f, label = "Truth",title = "i = $i")
    plot!(x, P*mu_g, label = "Gauss", color = colors[1])
    scatter!(Z, mu_g, label = "Gauss", color = colors[1])
    plot!(x, P*mu_a, label = "ADVI", color = colors[2])
    scatter!(Z, mu_a, label = "ADVI", color = colors[2])
    plot!(x, P*mu_s, label = "Stein", color = colors[3])
    scatter!(Z, mu_s, label = "Stein", color = colors[3])
end

display(g)

labels = ["Gauss" "ADVI" "Stein"]
plot(get.([g_h, a_h, s_h], :l_kernel), label = labels)
plot(get.([g_h, a_h, s_h], :σ_kernel), label = labels)
plot(get.([g_h, a_h, s_h], :σ_gaussian), label = labels)

using DataFrames
function Base.convert(::Type{DataFrame}, h::MVHistory)
    names = collect(keys(h))
    values = map(names) do key
        ValueHistories.values(h[key]).values
    end
    return DataFrame(values, names)
end
convert(DataFrame, a_h)
values(g_h[:mu]).values
collect(keys(a_h))
