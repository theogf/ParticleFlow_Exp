include(srcdir("train_model.jl"))
function run_SWAG(exp_p)
    n_iters = exp_p[:n_iters]
    n_runs = exp_p[:n_runs]
    AVI.setadbackend(:zygote)
    dim = exp_p[:dim]
    n_particles = exp_p[:n_particles]
    n_period = exp_p[:n_period]
    Random.seed!(exp_p[:seed])
    device = exp_p[:device]
    model = exp_p[:model]
    layers = exp_p[:layers]
    data = exp_p[:model]

    ## Loading the model and creating the appropriate function
    modelfile = projectdir("bnn_models", model, "model.bson")
    m = BSON.@load modelfile model
    fixed_m = m[1:(first(layers) - 1)] |> device
    opt_m = m[layers]
    opt_θ, opt_re = Flux.destructure(opt_m)
    n_θ = length(dense_θ)
    opt_m_sizes = broadcast(x-> length(x.W) + length(x.b), densem)
    nn_id_layers = vcat(0, cumsum(densem_sizes))

    ## Rebuild model given parameters and run it on data
    function nn_forward(xs, θ)
        opt_m = opt_re(θ)
        nn = Chain(fixed_m, opt_m)
        return nn(xs)
    end

    ## Define prior
    α = exp_p[:α]
    Flux.@functor TuringDiagMvNormal
    prior_θ = TuringDiagMvNormal(zeros(n_θ), α .* ones(n_θ)) |> device

    ## Define list of arguments and load the data
    args = Args()
    train_loader, test_loader = get_data(args)
    n_data = train_loader.imax
    n_batch = length(train_loader)

    ## Define the function to optimize
    function meta_logjoint(dummy)
        xs, ys = Random.nth(train_loader, rand(1:n_batch)) |> device
        return function logjoint(θ)
            logprior = logpdf(prior_θ, θ)
            pred = nn_forward(xs, θ)
            loglike = - n_data * Flux.logitcrossentropy(pred, ys)
            return logprior - negloglike
        end
    end


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
            :cond1 => exp_p[:cond1],
            :cond2 => exp_p[:cond2],
            :opt => deepcopy(opt),
            :callback => wrap_cb(),
            :mf => false,
            :init => x_init,
        )
        advi_p = Dict(
            :run => exp_p[:advi],
            :n_samples => n_particles,
            :max_iters => n_iters,
            :opt => deepcopy(opt),
            :callback => wrap_cb(),
            :init => (μ_init, sqrt.(Σ_init)),
        )
        stein_p = Dict(
            :run => exp_p[:steinvi],
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
