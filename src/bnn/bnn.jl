include(srcdir("train_model.jl"))
include(srcdir("utils", "bnn.jl"))
include(srcdir("utils", "callback.jl"))
include(srcdir("utils.jl"))

function run_bnn(exp_p)
    @unpack seed = exp_p
    Random.seed!(seed)

    ## Loading the model and creating the appropriate function
    @unpack use_gpu, dataset, activation, n_hidden, batchsize = exp_p # Load all variables from the dict exp_p
    device = use_gpu ? gpu : cpu
    model = "BNN_" * @savename(activation, n_hidden)
    modelfile = datadir("exp_raw", "bnn", "models", dataset, @savename(activation, n_hidden) * ".bson")
    m = BSON.load(modelfile)[:nn] |> device

    AVI.setadbackend(:zygote)
    ## Loading parameters for GPF
    @unpack L, n_iter, opt_det, opt_stoch, eta, α = exp_p
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
    nn_id_layers = vcat(0, cumsum(m_sizes))
    ## Loading specific parameters to GPF
    @unpack natmu, σ_init, mf = exp_p
    if natmu && !(exp_p[:alg] ∈ [:gf, :gpf])
        warn("Natural gradient are not compatible with non-Gaussian flows")
        return nothing
    end
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
        string(exp_p[:alg]) * "_" * model,
        @savename L n_iter batchsize mf σ_init natmu α opt_det opt_stoch
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
    function meta_logjoint(::Any)
        xs, ys = Random.nth(train_loader, rand(1:n_batch)) |> device
        return function logjoint(θ)
            # Computing logprior (this is kept fixed)
            logprior = logpdf(α, θ)
            # Making prediction on minibatch
            pred = nn_forward(reshape(xs, 28 * 28, :), θ)
            # Scaled up loglikelihood (logitcrossentropy returns a mean)
            loglike = -n_data * Flux.logitcrossentropy(pred, ys)
            return logprior + loglike
        end
    end
    x_init = device(randn(n_θ, n_particles) * sqrt(σ_init)) .+ θ
    Σ_init = device(Diagonal{Float32}(σ_init * I(length(θ))))

    ## Create dictionnaries of parameters
    general_p = Dict(
        :hyper_params => [],
        :hp_optimizer => nothing,
        :n_dim => dim,
        :gpu => device == gpu,
    )
    params = Dict{Symbol, Dict}()
    if exp_p[:alg] == :gpf
        params[:gpf] = Dict(
            :run => exp_p[:alg] == :gpf,
            :n_particles => n_particles,
            :max_iters => n_iter,
            :natmu => natmu,
            :opt => @eval($opt_det($eta)),
            :callback => wrap_heavy_cb(path=save_path),
            :init => copy(x_init),
            :mf => mf_option,
            :init => x_init,
            :gpu => device == gpu,
        )
    else
        params[:gpf] = no_run
    end
    if exp_p[:alg] == :gf
        params[:gf] = Dict(
            :run => exp_p[:alg] == :gf,
            :n_samples => n_particles,
            :max_iters => n_iter,
            :natmu => natmu,
            :opt => @eval($opt_stoch($eta)),
            :callback =>wrap_heavy_cb(path=save_path),
            :mf => mf_option,
            :gpu => device == gpu,
            :init => (mf == :full ? (copy(θ), sqrt.(diag(Σ_init))) : (copy(θ), cov_to_lowrank(Σ_init, L))) |> device,
        )
    else
        params[:gf] = no_run
    end
    if exp_p[:alg] == :dsvi
        params[:dsvi] = Dict(
            :run => exp_p[:mf] == :full ? (exp_p[:alg] == :dsvi) : false,
            :n_samples => n_particles,
            :max_iters => n_iter,
            :opt => @eval($opt_stoch($eta)),
            :callback => wrap_heavy_cb(path=save_path),
            :mf => Inf,
            :init => (copy(θ), copy(diag(Σ_init))),
            :gpu => device == gpu,
        )
    else
        params[:dsvi] = no_run
    end
    if exp_p[:alg] == :fcs
        params[:fcs] = Dict(
            :run => exp_p[:mf] == :none ? (exp_p[:alg] == :fcs) : false,
            :n_samples => n_particles,
            :max_iters => n_iter,
            :opt => @eval($opt_stoch($eta)),
            :callback => wrap_heavy_cb(path=save_path),
            :mf => false,
            :init => (copy(θ), cov_to_lowrank_plus_diag(Σ_init, L)...),
            :gpu => device == gpu,
        )
    else
        params[:fcs] = no_run
    end
    if exp_p[:alg] == :svgd_linear
        params[:svgd_linear] = Dict(
            :run => exp_p[:mf] == :none ? (exp_p[:alg] == :svgd_linear) : false,
            :n_particles => n_particles,
            :max_iters => n_iter,
            :opt => @eval($opt_det($eta)),
            :callback => wrap_heavy_cb(path=save_path),
            :mf => false,
            :init => copy(x_init),
            :gpu => device == gpu,
        )
    else
        params[:svgd_linear] = no_run
    end
    if exp_p[:alg] == :svgd_rbf
        params[:svgd_rbf] = Dict(
            :run => exp_p[:mf] == :none ? (exp_p[:alg] == :svgd_rbf) : false,
            :n_particles => n_particles,
            :max_iters => n_iter,
            :opt => @eval($opt_det($eta)),
            :callback => wrap_heavy_cb(path=save_path),
            :mf => false,
            :init => copy(x_init),
            :gpu => device == gpu,
        )
    else
        params[:svgd_rbf] = no_run
    end
    params[:iblr] = no_run
    @info "Initialized algs"
    # Train all models
    train_model(meta_logjoint, general_p, params)
    return save_path

end
