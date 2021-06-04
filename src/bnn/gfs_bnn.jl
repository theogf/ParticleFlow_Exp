include(srcdir("train_model.jl"))
include(srcdir("utils", "bnn.jl"))
include(srcdir("utils", "callback.jl"))

function run_gfs_bnn(exp_p)
    @unpack seed = exp_p
    Random.seed!(seed)

    ## Loading the model and creating the appropriate function
    @unpack use_gpu, model, dataset, activation, n_hidden, batchsize = exp_p # Load all variables from the dict exp_p
    device = use_gpu ? gpu : cpu
    modelfile = datadir("exp_raw", "bnn", "models", dataset, @savename(activation, n_hidden) * ".bson")
    m = BSON.load(modelfile)[:nn] |> device

    AVI.setadbackend(:zygote)
    ## Loading parameters for GPF
    @unpack L, n_iter, opt, α = exp_p
    n_particles = L + 1
    θ, re = Flux.destructure(m)
    n_θ = length(θ)
    m_sizes = broadcast(m) do x
        p = Flux.params(x)
        if length(p) == 0
            0
        else
            sum(length, p)
        end
    end
    nn_id_layers = vcat(0, cumsum(opt_m_sizes))

    ## Loading specific parameters to GPF
    @unpack cond1, cond2, σ_init, mf = exp_p
    σ_init = rand(MvNormal(θ, Float32(σ_init)), n_particles)

    mf_option = if mf == :partial
        nn_id_layers
    elseif mf == :full
        Inf
    elseif mf == :none
        false
    else
        error("Wrong option for mf")
    end
    ## Rebuild model given parameters and run it on data
    function nn_forward(xs, θ)
        nn = re(θ)
        return nn(xs)
    end

    save_path = datadir(
        "results",
        "bnn",
        dataset,
        "GPF_" * model,
        @savename L n_iter batchsize mf σ_init cond1 cond2 α
    )

    ## Define prior
    function logpdf(α::Real, θ::AbstractVector)
        - sum(abs2, θ) / (2α^2)
    end
    ## Define list of arguments and load the data
    train_loader, test_loader = get_data(dataset, batchsize)
    n_data = train_loader.imax
    n_batch = length(train_loader)

    ## Define the function to optimize
    function meta_logjoint(dummy)
        xs, ys = Random.nth(train_loader, rand(1:n_batch)) |> device
        return function logjoint(θ)
            # Computing logprior (this is kept fixed)
            logprior = logpdf(α, θ)
            # Making prediction on minibatch
            pred = nn_forward(xs, θ)
            # Scaled up loglikelihood (logitcrossentropy returns a mean)
            loglike = -n_data * Flux.logitcrossentropy(pred, ys)
            return logprior + loglike
        end
    end

    x_init = gpu(randn(n_θ, n_particles)) .+ opt_θ
    ## Create dictionnaries of parameters
    general_p = Dict(
        :hyper_params => [],
        :hp_optimizer => nothing,
        :n_dim => dim,
        :gpu => device == gpu,
    )
    gpf_p = Dict(
        :run => true,
        :n_particles => n_particles,
        :max_iters => n_iter,
        :cond1 => cond1,
        :cond2 => cond2,
        :opt => deepcopy(opt),
        :callback => wrap_heavy_cb(path = save_path),
        :mf => mf_option,
        :init => x_init,
        :gpu => device == gpu,
    )
    gf_p = Dict(
        :run => true,
        :n_particles => n_particles,
        :max_iters => n_iter,
        :cond1 => cond1,
        :cond2 => cond2,
        :opt => deepcopy(opt),
        :callback => wrap_heavy_cb(path = save_path),
        :mf => mf_option,
        :init => x_init,
        :gpu => device == gpu,
    )
    # Train all models
    train_model(meta_logjoint, general_p, gpf_p, gf_p, no_run)
end
