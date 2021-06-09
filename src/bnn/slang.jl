include(srcdir("utils", "bnn.jl"))
include(srcdir("algs", "slang.jl"))
function run_slang(exp_p)
    @unpack seed = exp_p
    Random.seed!(seed)

    ## Loading the model and creating the appropriate function
    @unpack use_gpu, dataset, activation, n_hidden, batchsize = exp_p # Load all variables from the dict exp_p
    device = use_gpu ? gpu : cpu
    model = "BNN_" * @savename(activation, n_hidden)
    modelfile = datadir("exp_raw", "bnn", "models", dataset, @savename(activation, n_hidden) * ".bson")
    m = BSON.load(modelfile)[:nn] |> device
    θ, re = Flux.destructure(m)

    ## Loading parameters for SLANG
    @unpack L, alpha, beta, α = exp_p
    save_path = datadir("results", "bnn", dataset, "slang_" * model, @savename L batchsize α alpha beta)
    ispath(save_path) ? nothing : mkpath(save_path)

    function save_params(alg::SLANG, i)
        parameters = [cpu(alg.μ), cpu(alg.U), cpu(alg.d)]
        tagsave(joinpath(save_path, savename(@dict i) * ".bson"), @dict parameters i)
    end
    ## Define prior
    logprior(θ::AbstractArray{<:Real}) = sum(abs2, θ)
    logprior(θ, α) = - sum(logprior, θ) / α^2

    loss(ŷ, y) = Flux.Losses.logitcrossentropy(ŷ, y)
    function meta_loss(x, y)
        return function(nn)
            loss(nn(θ)(x), y)
        end
    end
    ## Define list of arguments and load the data
    train_loader, test_loader = get_data(dataset, batchsize)
    iter = 0
    @functor SLANG
    alg = SLANG(L, length(θ), alpha, beta, α) |> device
    save_params(alg, 0)
    for _ in 1:n_epoch
        # p = ProgressMeter.Progress(length(train_loader))
        for (x, y) in train_loader
            x, y = x |> device, y |> device
            step!(alg, re, meta_loss(x, y))
            iter += 1
            if mod(iter, 250) == 0
                @info "Saving at iteration $iter"
                save_params(alg, iter)
            end
        end
    end
end
