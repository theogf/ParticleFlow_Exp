include(srcdir("train_model.jl"))
include(srcdir("utils", "bnn.jl"))
include(srcdir("utils", "callback.jl"))

function run_advi_bnn(exp_p)
    @unpack seed = exp_p
    Random.seed!(seed)

    ## Loading the model and creating the appropriate function
    @unpack use_gpu, model, dataset, batchsize = exp_p # Load all variables from the dict exp_p
    device = use_gpu ? gpu : cpu
    modelfile = projectdir("bnn_models", model, "model.bson")
    m = BSON.load(modelfile)[:model] |> device

    AVI.setadbackend(:zygote)
    ## Loading parameters for GPF
    @unpack start_layer, n_particles, n_iter, opt, α = exp_p
    fixed_m = m[1:(start_layer-1)] |> device
    opt_m = m[start_layer:end]
    opt_θ, opt_re = Flux.destructure(opt_m)
    n_θ = length(opt_θ)
    opt_m_sizes = broadcast(opt_m) do x
        p = Flux.params(x)
        if length(p) == 0
            0
        else
            sum(length, p)
        end
    end

    ## Rebuild model given parameters and run it on data
    function nn_forward(xs, θ)
        opt_m = opt_re(θ)
        nn = Chain(fixed_m, opt_m)
        return nn(xs)
    end

    save_path = datadir(
        "results",
        "bnn",
        dataset,
        "ADVI_" * model,
        @savename n_particles start_layer n_iter batchsize α
    )

    ## Define prior
    Flux.@functor TuringDiagMvNormal
    prior_θ = TuringDiagMvNormal(zeros(n_θ), α .* ones(n_θ)) |> device

    ## Define list of arguments and load the data
    train_loader, test_loader = get_data(dataset, batchsize)
    n_data = train_loader.imax
    n_batch = length(train_loader)

    ## Define the function to optimize
    function meta_logjoint(dummy)
        xs, ys = Random.nth(train_loader, rand(1:n_batch)) |> device
        return function logjoint(θ)
            # Computing logprior (this is kept fixed)
            logprior = logpdf(prior_θ, θ)
            # Making prediction on minibatch
            pred = nn_forward(xs, θ)
            # Scaled up loglikelihood (logitcrossentropy returns a mean)
            loglike = -n_data * Flux.logitcrossentropy(pred, ys)
            return logprior + loglike
        end
    end
    ## Create dictionnaries of parameters
    general_p = Dict(
        :hyper_params => [],
        :hp_optimizer => nothing,
        :n_dim => dim,
        :gpu => device == gpu,
    )
    advi_p = Dict(
        :run => true,
        :n_samples => n_particles,
        :max_iters => n_iter,
        :opt => deepcopy(opt),
        :callback => wrap_heavy_cb(path = save_path),
        :mf => Inf,
        :init => (opt_θ, ones(n_θ)),
        :gpu => device == gpu,
    )
    # Train all models
    train_model(meta_logjoint, general_p, no_run, advi_p, no_run)
end
