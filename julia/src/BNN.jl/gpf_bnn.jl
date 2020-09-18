include(srcdir("train_model.jl"))
include(srcdir("utils", "bnn.jl"))

function run_gpf_bnn(exp_p)
    @unpack seed = exp_p
    Random.seed!(seed)

    ## Loading the model and creating the appropriate function
    @unpack use_gpu, model, dataset, batchsize, η = exp_p # Load all variables from the dict exp_p
    device = use_gpu ? gpu : cpu
    modelfile = projectdir("bnn_models", model, "model.bson")
    m = (BSON.@load modelfile model) |> device

    AVI.setadbackend(:zygote)
    ## Loading parameters for GPF
    @unpack layers, n_particles, n_iters, opt = exp_p
    fixed_m = m[1:(first(layers)-1)] |> device
    opt_m = m[layers]
    opt_θ, opt_re = Flux.destructure(opt_m)
    n_θ = length(dense_θ)
    opt_m_sizes = broadcast(x -> length(x.W) + length(x.b), densem)
    nn_id_layers = vcat(0, cumsum(densem_sizes))

    ## Loading specific parameters to GPF
    @unpack cond1, cond2, σ_init, mf = exp_p
    σ_init = rand(MvNormal(opt_θ, Float32(σ_init)), n_particles)
    mf_option = if mf == :layers
        nn_id_layers
    elseif mf == :ind
        0:n_θ
    elseif mf == :full
        false
    else
        error("Wrong option for mf")
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
        "GPF",
        @savename n_particles n_iter batchsize mf σ_init cond1 cond2
    )

    ## Define prior
    @unpack α = exp_p
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
            negloglike = -n_data * Flux.logitcrossentropy(pred, ys)
            return logprior - negloglike
        end
    end


    ## Create dictionnaries of parameters
    general_p = Dict(
        :hyper_params => [],
        :hp_optimizer => nothing,
        :n_dim => dim,
        :gpu => device == gpu,
    )
    gflow_p = Dict(
        :run => exp_p[:gpf],
        :n_particles => n_particles,
        :max_iters => n_iters,
        :cond1 => cond1,
        :cond2 => cond2,
        :opt => deepcopy(opt),
        :callback => wrap_heavy_cb(path = save_path),
        :mf => nn_id_layers,
        :init => x_init,
        :gpu => device == gpu,
    )
    no_run = Dict(:run => false)
    # Train all models
    train_model(meta_logjoint, general_p, gflow_p, no_run, no_run)
end
