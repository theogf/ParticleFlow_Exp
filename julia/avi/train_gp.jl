include("train_model.jl")

n_samples = 60
n_iters = 10

## Create some toy data
N = 50
x = range(0, 1, length = N)
θ = log.([1.0, 10.0, 1e-3])
k = exp(θ[1]) * transform(SqExponentialKernel(), exp(θ[2]))
K = kernelmatrix(k, x) + 1e-5I
f = rand(MvNormal(K))
y = f + randn(N) * exp(θ[3])
plot(x, y)

## Create the model
function meta_logπ(θ)
    k = exp(θ[1]) * transform(SqExponentialKernel(), exp(θ[2]))
    K = kernelmatrix(k, x) + 1e-5I
    d = TuringDenseMvNormal(zeros(length(x)), K)
    return z -> sum(logpdf.(Normal.(y, exp(θ[3])), z)) + logpdf(d, z)
end

logπ_reduce = meta_logπ(θ)
AVI.setadbackend(:reversediff)
##
hp_init = θ .- 1

opt = ADAGrad(1.0)

general_p =
    Dict(:hyper_params => hp_init, :hp_optimizer => ADAGrad(0.1), :n_dim => N)
gflow_p = Dict(
    :run => true,
    :n_particles => 60,
    :max_iters => n_iters,
    :cond1 => true,
    :cond2 => false,
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
    # train_model(x, y, logπ_reduce, general_p, gflow_p, advi_p, stein_p)
    train_model(x, y, meta_logπ, general_p, gflow_p, advi_p, stein_p)

## Plotting

iters, mus_g = get(g_h, :mu)
iters, mus_a = get(a_h, :mu)
iters, mus_s = get(s_h, :mu)

@gif for (i, mu_g, mu_a, mu_s) in zip(iters, mus_g, mus_a, mus_s)
    plot(x, f, label = "Truth",title = "i = $i")
    plot!(x, mu_g, label = "Gauss",)
    plot!(x, mu_a, label = "ADVI")
    plot!(x, mu_s, label = "Stein")
end

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
