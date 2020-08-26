include("train_model.jl")

n_iters = 50
# AVI.setadbackend(:forwarddiff)
AVI.setadbackend(:reversediff)
## Create target distribution
D = 200
n_samples = D + 1
μ = sort(randn(D))
Σ = I(D)
Σ = rand(D, D) |> x -> x * x'
d_target = MvNormal(μ, Σ)

## Create the model
function logπ(θ)
    return logpdf(d_target, θ)
end

##

opt = ADAGrad(1.0)

general_p =
    Dict(:hyper_params => nothing, :hp_optimizer => ADAGrad(0.1), :n_dim => D)
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
    :run => true,
    :n_particles => n_samples,
    :max_iters => n_iters,
    :kernel => transform(SqExponentialKernel(), 1.0),
    :opt => deepcopy(opt),
    :callback => wrap_cb,
    :init => nothing,
)


g_h, a_h, s_h =
    train_model(logπ, general_p, gflow_p, advi_p, stein_p)
    # train_model(x, y, meta_logπ, general_p, gflow_p, advi_p, stein_p)

## Plotting erros


err_μ_g = norm.(Ref(μ) .- get(g_h, :mu)[2])
p_μ = plot(err_μ_g, label = "Gauss", xlabel = "t", ylabel = "|μ - μ'|'", yaxis = :log, color = colors[1])
if stein_p[:run]
    err_μ_s = norm.(Ref(μ) .- get(s_h, :mu)[2])
    plot!(err_μ_s, label = "Stein", color = colors[3])
end
if advi_p[:run]
    err_μ_a = norm.(Ref(μ) .- get(a_h, :mu)[2])
    plot!(err_μ_a, label = "ADVI", color = colors[2])
end
display(p_μ)
savefig(joinpath(@__DIR__, "..", "plots", "gaussian", "D=$(D)_" * (d_target isa DiagNormal ? "diag" : "") * ".png"))

err_Σ_g = norm.(Ref(Σ) .- reshape.(get(g_h, :sig)[2], D, D))
p_Σ = plot(err_μ_g, label = "Gauss", xlabel = "t", ylabel = "|Σ - Σ'|'", yaxis = :log, color = colors[1])
if stein_p[:run]
    err_Σ_s = norm.(Ref(Σ) .- reshape.(get(s_h, :sig)[2], D, D))
    plot!(err_Σ_s, label = "Stein", color = colors[3])
end
if advi_p[:run]
    err_Σ_a = norm.(Ref(Σ) .- reshape.(get(a_h, :sig)[2], D, D))
    plot!(err_Σ_a, label = "ADVI", color = colors[2])
end
display(p_Σ)


## Plotting the evolution of the means

μs = []
iters = get(g_h, :mu)[1]
isempty(g_h.storage) ? nothing : push!(μs, get(g_h, :mu)[2])
# isempty(a_h.storage) ? nothing : push!(μs, get(a_h, :mu)[2])
isempty(s_h.storage) ? nothing : push!(μs, get(s_h, :mu)[2])
Σs = []
isempty(g_h.storage) ? nothing : push!(Σs, reshape.(get(g_h, :sig)[2], D, D))
# isempty(a_h.storage) ? nothing : push!(Σs, reshape.(get(a_h, :sig)[2], D, D))
isempty(s_h.storage) ? nothing : push!(Σs, reshape.(get(s_h, :sig)[2], D, D))


# g = @gif for (i, mu_g, sig_g) in zip(iters, mus...,Σs...)
# g = @gif for (i, mu_g, mu_a, mu_s, sig_g, sig_a, sig_s) in zip(iters, μs..., Σs...)
g = @gif for (i, mu_g, mu_s, sig_g, sig_s) in zip(iters, μs..., Σs...)
    p1 = plot(μ, label = "Truth",title = "μ : i = $(i)", color=colors[4])
    if length(μs) == 1
        plot!(mu_g, label = "Gauss", color=colors[1])
    elseif length(μs) == 2
        plot!(mu_g, label = "Gauss", color=colors[1])
        plot!(mu_s, label = "Stein", color=colors[3])
    else
        plot!(mu_g, label = "Gauss", color=colors[1])
        plot!(mu_s, label = "Stein", color=colors[3])
        plot!(mu_a, label = "ADVI", color=colors[2])
    end
    p2 = plot(diag(Σ), label = "Truth", title = "diag(Σ) : i = $(i)", color=colors[4])
    if length(μs) == 1
        plot!(diag(sig_g), label = "Gauss", color=colors[1])
    elseif length(μs) == 2
        plot!(diag(sig_g), label = "Gauss", color=colors[1])
        plot!(diag(sig_s), label = "Stein", color=colors[3])
    else
        plot!(diag(sig_g), label = "Gauss", color=colors[1])
        plot!(diag(sig_s), label = "Stein", color=colors[3])
        plot!(diag(sig_a), label = "ADVI", color=colors[2])
    end
    plot(p1, p2)
end

display(g)
## More
# labels = ["Gauss" "ADVI" "Stein"]
# # plot(get.([g_h, a_h, s_h], :l_kernel), label = labels)
# # plot(get.([g_h, a_h, s_h], :σ_kernel), label = labels)
# # plot(get.([g_h, a_h, s_h], :σ_gaussian), label = labels)
#
# using DataFrames
# function Base.convert(::Type{DataFrame}, h::MVHistory)
#     names = collect(keys(h))
#     values = map(names) do key
#         ValueHistories.values(h[key]).values
#     end
#     return DataFrame(values, names)
# end
# convert(DataFrame, a_h)
# values(g_h[:mu]).values
# collect(keys(a_h))
