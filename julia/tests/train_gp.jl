include("train_model.jl")

n_samples = 100
n_iters = 150
AVI.setadbackend(:reversediff)
# AVI.setadbackend(:forwarddiff)
## Create some toy data
N = 500
x = range(0, 1, length = N)
θ = log.([1.0, 10.0, 1e-3])
k = exp(θ[1]) * transform(SqExponentialKernel(), exp(θ[2]))
K = kernelmatrix(k, x) + 1e-3I
f = rand(MvNormal(K))
likelihood(f, θ) = Normal(f, sqrt(exp(θ[3])))
y = rand.(likelihood.(f, Ref(θ)))
# likelihood(f, θ) = Bernoulli(exp(logsigmoid(f)))
# y = (sign.(f) .+ 1) .÷ 2
scatter(x, y) |> display

## Create the model
function meta_logπ(θ)
    k = exp(θ[1]) * transform(SqExponentialKernel(), exp(θ[2]))
    K = kernelmatrix(k, x) + 1e-5I
    d = TuringDenseMvNormal(zeros(length(x)), K)
    return z -> sum(logpdf.(likelihood.(z, Ref(θ)), y)) + logpdf(d, z)
end

logπ_reduce = meta_logπ(θ)
# logπ_reduce(mus_g[end])

##
hp_init = θ .- 1

opt = ADAGrad(1.0)

general_p =
    Dict(:hyper_params => nothing, :hp_optimizer => ADAGrad(0.1), :n_dim => N)
gflow_p = Dict(
    :run => true,
    :n_particles => 60,
    :max_iters => n_iters,
    :cond1 => false,
    :cond2 => false,
    :opt => deepcopy(opt),
    :callback => wrap_cb,
    :init => nothing,
)
advi_p = Dict(
    :run => !true,
    :n_samples => n_samples,
    :max_iters => n_iters,
    :opt => deepcopy(opt),
    :callback => wrap_cb,
    :init => nothing,
)
stein_p = Dict(
    :run => !true,
    :n_particles => n_samples,
    :max_iters => n_iters,
    :kernel => transform(SqExponentialKernel(), 1.0),
    :opt => deepcopy(opt),
    :callback => wrap_cb,
    :init => nothing,
)


g_h, a_h, s_h =
    train_model(logπ_reduce, general_p, gflow_p, advi_p, stein_p)
    # train_model(x, y, meta_logπ, general_p, gflow_p, advi_p, stein_p)

## Plotting

μs = []
iters = get(g_h, :mu)[1]
isempty(g_h.storage) ? nothing : push!(μs, get(g_h, :mu)[2])
# isempty(a_h.storage) ? nothing : push!(μs, get(a_h, :mu)[2])
# isempty(s_h.storage) ? nothing : push!(μs, get(s_h, :mu)[2])
Σs = []
isempty(g_h.storage) ? nothing : push!(Σs, reshape.(get(g_h, :sig)[2], N, N))
# isempty(a_h.storage) ? nothing : push!(Σs, reshape.(get(a_h, :sig)[2], N, N))
# isempty(s_h.storage) ? nothing : push!(Σs, reshape.(get(s_h, :sig)[2], N, N))


a = @animate for (i, mu_g, sig_g) in zip(iters, μs..., Σs...)
# g = @gif for (i, mu_g, mu_a, mu_s) in zip(iters, mus...)
    plot(x, f, label = "Truth",title = "i = $(i)", color=colors[4])
    scatter!(x, y, label = "Data", color=colors[4])
    if length(mus) == 1
        plot!(x, mu_g, ribbon = sqrt.(diag(sig_g)), label = "Gauss", color=colors[1])
    else
        plot!(x, mu_g, label = "Gauss", color=colors[1])
        plot!(x, mu_s, label = "Stein", color=colors[3])
        plot!(x, mu_a, label = "ADVI", color=colors[2])
    end
end
gif(a, joinpath(@__DIR__, "..", "plots", "gifs", "conv_gp.gif"), fps = 10)

## Plotting for evolution of hyperparameters
labels = ["Gauss" "ADVI" "Stein"]
# plot(get.([g_h, a_h, s_h], :l_kernel), label = labels)
# plot(get.([g_h, a_h, s_h], :σ_kernel), label = labels)
# plot(get.([g_h, a_h, s_h], :σ_gaussian), label = labels)

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
