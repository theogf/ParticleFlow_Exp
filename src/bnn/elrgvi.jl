include(srcdir("utils", "bnn.jl"))
include(srcdir("algs", "elrgvi.jl"))
function run_elrgvi(exp_p)
    @unpack seed = exp_p
    Random.seed!(seed)

    ## Loading the model and creating the appropriate function
    @unpack use_gpu, n_iter, dataset, activation, n_hidden, batchsize = exp_p # Load all variables from the dict exp_p
    device = use_gpu ? gpu : cpu
    model = "BNN_" * @savename(activation, n_hidden)
    modelfile = datadir("exp_raw", "bnn", "models", dataset, @savename(activation, n_hidden) * ".bson")
    m = BSON.load(modelfile)[:nn] |> device

    ## Loading parameters for SLANG
    @unpack L, opt, eta, α = exp_p
    save_path = datadir("results", "bnn", dataset, "slang_" * model, @savename L n_iter batchsize α eta opt)
    ispath(save_path) ? nothing : mkpath(save_path)

    function save_params(alg::Chain, i)
        parameters = save_params.(alg)
        tagsave(joinpath(save_path, savename(@dict i) * ".bson"), @dict parameters i)
    end
    function save_params(alg::LowRankGaussianDenseLayer)
        return [cpu(alg.μ), cpu(alg.v), cpu(alg.σ)]
    end
    ## Define prior
    logprior(θ::AbstractArray{<:Real}) = sum(abs2, θ)
    logprior(θ, α) = - sum(logprior, θ) / α^2

    loss(ŷ, y) = Flux.Losses.logitcrossentropy(ŷ, y)
    function neg_elbo(l, x, y)
        return n_data / batchsize * loss(l(reshape(x, 28 * 28, :)), y) + LRKLdivergence(l, α)
    end
    ## Define list of arguments and load the data
    iter = 0
    train_loader, test_loader = get_data(dataset, batchsize)
    n_data = train_loader.imax
    n_batch = length(train_loader)
    m = Chain(LowRankGaussianDenseLayer.(m, L)...) |> device
    ps = Flux.params(m)
    save_params(m, 0)
    x, y = first(train_loader) |> device
    neg_elbo(m, x, y)
    @showprogress for _ in 1:n_iter
        x, y = Random.nth(train_loader, rand(1:n_batch)) |> device
        grads = gradient(ps) do 
            neg_elbo(m, x, y)
        end
        iter += 1
        Flux.Optimise.update!(opt, ps, grads)
                if mod(iter, 250) == 0
            @info "Saving at iteration $iter"
            save_params(m, iter)
        end
    end
end
