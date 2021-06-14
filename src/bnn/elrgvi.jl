include(srcdir("utils", "bnn.jl"))
include(srcdir("algs", "elrgvi.jl"))
include(srcdir("utils.jl"))
function run_elrgvi(exp_p)
    @unpack seed = exp_p
    Random.seed!(seed)

    ## Loading the model and creating the appropriate function
    @unpack use_gpu, n_iter, dataset, activation, n_hidden, batchsize = exp_p # Load all variables from the dict exp_p
    device = use_gpu ? gpu : cpu
    global DEFAULT_RNG = use_gpu ? CUDA.CURAND.default_rng() : Random.GLOBAL_RNG
    model = "BNN_" * @savename(activation, n_hidden)
    modelfile = datadir("exp_raw", "bnn", "models", dataset, @savename(activation, n_hidden) * ".bson")
    m = BSON.load(modelfile)[:nn] |> device

    ## Loading parameters for SLANG
    @unpack L, opt, eta, α = exp_p
    save_path = datadir("results", 
                        "bnn",
                        dataset,
                        model, 
                        "elrgvi",
                        @savename L n_iter batchsize α eta opt)
    ispath(save_path) ? nothing : mkpath(save_path)
    function save_params(alg::Chain, i)
        parameters = save_params.(alg)
        tagsave(joinpath(save_path, savename(@dict i) * ".bson"), @dict parameters i)
    end
    function save_params(alg::LowRankGaussianDenseLayer)
        return [cpu(alg.μ), cpu(alg.v), cpu(alg.σ)]
    end

    ## Define list of arguments and load the data
    iter = 0
    train_loader, test_loader = get_data(dataset, batchsize)
    n_data = train_loader.imax
    n_batch = length(train_loader)
    m = Chain(LowRankGaussianDenseLayer.(m, L)...) |> device
    ps = Flux.params(m)
    ## Define prior
    logprior(θ::AbstractArray{<:Real}) = sum(abs2, θ)
    logprior(θ, α) = - sum(logprior, θ) / α^2

    loss(ŷ, y) = Flux.Losses.logitcrossentropy(ŷ, y)
    function neg_elbo(l, x, y)
        KL = LRKLdivergence(l, α)
        ml_loss = n_data / batchsize * loss(l(reshape(x, 28 * 28, :)), y)
        return KL + ml_loss
    end
    save_params(m, 0)
    x, y = first(train_loader) |> device
    neg_elbo(m, x, y)
    optimiser = @eval $(opt)($eta)
    @showprogress for _ in 1:n_iter
        x, y = Random.nth(train_loader, rand(1:n_batch)) |> device
        # @timeit to "Gradients" 
        grads = gradient(ps) do 
            neg_elbo(m, x, y)
        end
        iter += 1
        Flux.Optimise.update!(optimiser, ps, grads)
        if mod(iter, 250) == 0
            @info "Saving at iteration $iter"
            save_params(m, iter)
        end
    end
end
