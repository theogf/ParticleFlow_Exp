include(srcdir("train_model.jl"))
include(srcdir("utils", "tools.jl"))
function run_gaussian_target(exp_p)
    @unpack seed = exp_p
    Random.seed!(seed)
    AVI.setadbackend(:reversediff)

    ## Create target distribution
    @unpack dim, n_particles, n_iters, n_runs, cond1, cond2, full_cov = exp_p[:dim]
    μ = sort(randn(dim))
    Σ = if full_cov
        Q, _ = qr(rand(dim, dim)) # Create random unitary matrix
        Λ = Diagonal(10.0.^(3 * ((1:dim) .- 1) ./ dim ))
        Q * Λ * Q'
    else
        I(dim)
    end

    # Flux.@functor TuringDenseMvNormal
    d_target = TuringDenseMvNormal(μ, Σ)
    ## Create the model
    function logπ_gauss(θ)
        return logpdf(d_target, θ)
    end

    gpf = []
    advi = []
    steinvi = []

    for i in 1:n_runs
        @info "Run $i/$(n_runs)"
        opt = exp_p[:opt]
        μ_init = randn(dim)
        Σ_init = Diagonal(exp.(randn(dim)))
        p_init = MvNormal(μ_init, Σ_init)
        x_init = rand(p_init, n_particles)

        ## Create dictionnaries of parameters
        general_p =
            Dict(:hyper_params => nothing, :hp_optimizer => ADAGrad(0.1), :n_dim => dim, :gpu => false)
        gflow_p = Dict(
            :run => exp_p[:gpf],
            :n_particles => n_particles,
            :max_iters => n_iters,
            :cond1 => cond1,
            :cond2 => cond2,
            :opt => deepcopy(opt),
            :callback => wrap_cb(),
            :mf => false,
            :init => x_init,
        )
        advi_p = Dict(
            :run => exp_p[:advi] && !cond1 && !cond2,
            :n_samples => n_particles,
            :max_iters => n_iters,
            :opt => deepcopy(opt),
            :callback => wrap_cb(),
            :init => (μ_init, sqrt.(Σ_init)),
        )
        stein_p = Dict(
            :run => exp_p[:steinvi] && !cond1 && !cond2,
            :n_particles => n_particles,
            :max_iters => n_iters,
            :kernel => KernelFunctions.transform(SqExponentialKernel(), 1.0),
            :opt => deepcopy(opt),
            :callback => wrap_cb(),
            :init => x_init,
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
