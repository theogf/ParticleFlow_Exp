include(srcdir("train_model.jl"))
include(srcdir("utils", "tools.jl"))
function run_gp_gpf(exp_p)
    @unpack seed = exp_p
    Random.seed!(seed)
    AVI.setadbackend(:reversediff)

    ## Create target distribution
    @unpack dim, n_particles, n_iters, n_runs, cond1, cond2, full_cov = exp_p
    μ = sort(randn(dim))
    Σ = if full_cov
        Q, _ = qr(rand(dim, dim)) # Create random unitary matrix
        Λ = Diagonal(10.0.^(3 * ((1:dim) .- 1) ./ dim ))
        Q * Λ * Q'
    else
        I(dim)
    end

    gpf = []

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
