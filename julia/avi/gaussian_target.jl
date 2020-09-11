include("train_model.jl")
function run_gaussian_target(exp_p)
    n_iters = exp_p[:n_iters]
    n_runs = exp_p[:n_runs]
    AVI.setadbackend(:reversediff)
    # AVI.setadbackend(:forwarddiff)
    ## Create target distribution
    dim = exp_p[:dim]
    n_particles = exp_p[:n_particles]
    Random.seed!(exp_p[:seed])
    cond1 = exp_p[:cond1]
    cond2 = exp_p[:cond2]
    μ = sort(randn(dim))
    full_cov = exp_p[:full_cov]
    Σ = if full_cov
        Q, _ = qr(rand(dim, dim)) # Create random unitary matrix
        Λ = Diagonal(10.0.^(3 * ((1:dim) .- 1) ./ dim ))
        Q * Λ * Q'
    else
        I(dim)
    end
    d_target = MvNormal(μ, Σ)

    ## Create the model
    function logπ_gauss(θ)
        return logpdf(d_target, θ)
    end
    gpf = []
    advi = []
    steinvi = []

    for i in 1:n_runs
        @info "Run $i/$(n_runs)"
        opt = ADAGrad(1.0)

        ## Create dictionnaries of parameters
        general_p =
            Dict(:hyper_params => nothing, :hp_optimizer => ADAGrad(0.1), :n_dim => dim)
        gflow_p = Dict(
            :run => exp_p[:gpf],
            :n_particles => n_particles,
            :max_iters => n_iters,
            :cond1 => exp_p[:cond1],
            :cond2 => exp_p[:cond2],
            :opt => deepcopy(opt),
            :callback => wrap_cb,
            :init => nothing,
        )
        advi_p = Dict(
            :run => exp_p[:advi],
            :n_samples => n_particles,
            :max_iters => n_iters,
            :opt => deepcopy(opt),
            :callback => wrap_cb,
            :init => nothing,
        )
        stein_p = Dict(
            :run => exp_p[:steinvi],
            :n_particles => n_particles,
            :max_iters => n_iters,
            :kernel => transform(SqExponentialKernel(), 1.0),
            :opt => deepcopy(opt),
            :callback => wrap_cb,
            :init => nothing,
        )

        # Train all models
        _gpf, _advi, _steinvi =
            train_model(logπ_gauss, general_p, gflow_p, advi_p, stein_p)
        push!(gpf, _gpf)
        push!(advi, _advi)
        push!(steinvi, _steinvi)
    end

    file_prefix = @savename dim n_particles full_cov n_iters n_runs

    tagsave(datadir("results", "gaussian", file_prefix * ".bson"),
            @dict dim n_particles full_cov n_iters n_runs cond1 cond2 gpf advi steinvi exp_p d_target;
            safe=false, storepatch = false)
end

    ## Plotting errors

    ## Plotting the evolution of the means
    #
    # μs = []
    # iters = get(g_h, :mu)[1]
    # isempty(g_h.storage) ? nothing : push!(μs, get(g_h, :mu)[2])
    # # isempty(a_h.storage) ? nothing : push!(μs, get(a_h, :mu)[2])
    # isempty(s_h.storage) ? nothing : push!(μs, get(s_h, :mu)[2])
    # Σs = []
    # isempty(g_h.storage) ? nothing : push!(Σs, reshape.(get(g_h, :sig)[2], dim, dim))
    # # isempty(a_h.storage) ? nothing : push!(Σs, reshape.(get(a_h, :sig)[2], dim, dim))
    # isempty(s_h.storage) ? nothing : push!(Σs, reshape.(get(s_h, :sig)[2], dim, dim))
    #
    #
    # g = @gif for (i, mu_g, sig_g) in zip(iters, μs...,Σs...)
    # # g = @gif for (i, mu_g, mu_a, mu_s, sig_g, sig_a, sig_s) in zip(iters, μs..., Σs...)
    # # g = @gif for (i, mu_g, mu_s, sig_g, sig_s) in zip(iters, μs..., Σs...)
    #     p1 = plot(μ, label = "Truth",title = "μ : i = $(i)", color=colors[4])
    #     if length(μs) == 1
    #         plot!(mu_g, label = "Gauss", color=colors[1])
    #     elseif length(μs) == 2
    #         plot!(mu_g, label = "Gauss", color=colors[1])
    #         plot!(mu_s, label = "Stein", color=colors[3])
    #     else
    #         plot!(mu_g, label = "Gauss", color=colors[1])
    #         plot!(mu_s, label = "Stein", color=colors[3])
    #         plot!(mu_a, label = "ADVI", color=colors[2])
    #     end
    #     p2 = plot(diag(Σ), label = "Truth", title = "diag(Σ) : i = $(i)", color=colors[4])
    #     if length(μs) == 1
    #         plot!(diag(sig_g), label = "Gauss", color=colors[1])
    #     elseif length(μs) == 2
    #         plot!(diag(sig_g), label = "Gauss", color=colors[1])
    #         plot!(diag(sig_s), label = "Stein", color=colors[3])
    #     else
    #         plot!(diag(sig_g), label = "Gauss", color=colors[1])
    #         plot!(diag(sig_s), label = "Stein", color=colors[3])
    #         plot!(diag(sig_a), label = "ADVI", color=colors[2])
    #     end
    #     plot(p1, p2)
    # end
    #
    # display(g)
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
