include(srcdir("utils", "bnn.jl"))
function run_SWAG(exp_p)
    @unpack seed = exp_p
    Random.seed!(seed)

    ## Loading the model and creating the appropriate function
    @unpack use_gpu, dataset, activation, n_hidden, batchsize = exp_p # Load all variables from the dict exp_p
    device = use_gpu ? gpu : cpu
    model = "BNN_" * @savename(activation, n_hidden)
    modelfile = datadir("bnn_models", model, "model.bson")
    m = BSON.load(modelfile)[:nn] |> device

    ## Loading parameters for SWAG
    @unpack n_period, n_epoch, eta, α = exp_p
    save_path = datadir("results", "bnn", dataset, "swag_" * model, @savename n_epoch n_period batchsize α eta)
    ispath(save_path) ? nothing : mkpath(save_path)

    function save_params(ps, i)
        parameters = [cpu(p) for p in ps]
        tagsave(joinpath(save_path, savename(@dict i) * ".bson"), @dict parameters i)
    end
    ## Define prior
    logprior(θ::AbstractArray{<:Real}) = sum(abs2, θ)
    logprior(θ, α) = - sum(logprior, θ) / α^2

    loss(ŷ, y) = Flux.Losses.logitcrossentropy(ŷ, y)

    ## Define list of arguments and load the data
    train_loader, test_loader = get_data(dataset, batchsize)
    iter = 0
    save_params(ps, 0)
    for _ in 1:n_epoch
        # p = ProgressMeter.Progress(length(train_loader))
        for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                ŷ = m(x)
                loss(ŷ, y) - logprior(ps, α)
            end
            Flux.Optimise.update!(opt, ps, gs)
            # ProgressMeter.next!(p)   # comment out for no progress bar
            iter += 1
            if mod(iter, n_period) == 0
                @info "Saving at iteration $iter"
                save_params(ps, iter)
            end
        end
    end
end
