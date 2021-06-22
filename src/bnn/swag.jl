include(srcdir("utils", "bnn.jl"))
function run_SWAG(exp_p)
    @unpack seed = exp_p
    Random.seed!(seed)
    ## Loading the model and creating the appropriate function
    @unpack use_gpu, dataset, activation, n_hidden, batchsize = exp_p # Load all variables from the dict exp_p
    device = use_gpu ? gpu : cpu
    model = "BNN_" * @savename(activation, n_hidden)
    modelfile = datadir("exp_raw", "bnn", "models", dataset, @savename(activation, n_hidden) * ".bson")
    m = BSON.load(modelfile)[:nn] |> device
    ps = Flux.params(m)
    ## Loading parameters for SWAG
    @unpack n_period, n_epoch, eta, rho, α = exp_p
    save_path = datadir("results", "bnn", dataset, model, "swag", @savename n_epoch n_period batchsize α eta rho)
    ispath(save_path) ? nothing : mkpath(save_path)

    function save_params(ps, i)
        parameters = [cpu(p) for p in ps]
        tagsave(joinpath(save_path, savename(@dict i) * ".bson"), @dict parameters i)
    end
    ## Define prior
    normalizer(θ::AbstractArray{<:Real}) = sum(abs2, θ)
    normalizer(θ, α) = sum(logprior, θ) / (2α^2)


    ## Define list of arguments and load the data
    train_loader, test_loader = get_data(dataset, batchsize)
    @show n_data = size(train_loader.data[1], 3)
    loss(ŷ, y) = Flux.Losses.logitcrossentropy(ŷ, y)
    iter = 0
    opt = ADAM()
    ## First train the NN to a MAP over 100 epochs
    @info "Training NN"
    @showprogress for i in 1:100
        for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                ŷ = m(reshape(x, 28 * 28, :))
                loss(ŷ, y) + logprior(ps, α) / n_data
            end
            Flux.Optimise.update!(opt, ps, gs)
        end
        @info string("Epoch $i/100, test_loss : ", loss(m(reshape(device(test_loader.data[1]), 28 * 28, :)), device(test_loader.data[2])))
        @info string("Epoch $i/100, train_loss : ", loss(m(reshape(device(train_loader.data[1]), 28 * 28, :)), device(train_loader.data[2])))
    end
    ## Then start saving samples with SGD with learning rate 0.5
    save_params(ps, 0)
    @info "Start sampling"
    sgd_opt = Momentum(eta, rho)
    for _ in 1:n_epoch
        # p = ProgressMeter.Progress(length(train_loader))
        for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                ŷ = m(reshape(x, 28 * 28, :))
                loss(ŷ, y) - logprior(ps, α)
            end
            Flux.Optimise.update!(sgd_opt, ps, gs)
            # ProgressMeter.next!(p)   # comment out for no progress bar
            iter += 1
            if mod(iter, n_period) == 0
                @info "Saving at iteration $iter"
                save_params(ps, iter)
            end
        end
    end
end
