include(srcdir("utils", "bnn.jl"))
function run_SWAG(exp_p)
    @unpack seed = exp_p
    Random.seed!(seed)

    ## Loading the model and creating the appropriate function
    @unpack use_gpu, model, dataset, batchsize, η = exp_p # Load all variables from the dict exp_p
    device = use_gpu ? gpu : cpu
    modelfile = projectdir("bnn_models", model, "model.bson")
    m = BSON.load(modelfile)[:model] |> device

    ## Loading parameters for SWAG
    @unpack start_layer, n_period, n_epoch = exp_p
    ps = Flux.params(m[start_layer:end])
    opt = Flux.Momentum(η)
    save_path = datadir("results", "bnn", dataset, "SWAG_" * model, @savename n_epoch n_period batchsize η start_layer)
    ispath(save_path) ? nothing : mkpath(save_path)

    function save_params(ps, i)
        path = save_path * "i=$(i)" * ".bson"
        parameters = [cpu(p) for p in ps]
        tagsave(joinpath(save_path, savename(@dict i) * ".bson"), @dict parameters i)
    end
    ## Define prior
    @unpack α = exp_p
    logprior(θ::AbstractArray{<:Real}) = sum(abs2, θ)
    logprior(θ, α) = - sum(logprior, θ) / α

    loss(ŷ, y) = Flux.Losses.logitcrossentropy(ŷ, y)

    ## Define list of arguments and load the data
    train_loader, test_loader = get_data(dataset, batchsize)
    iter = 0
    save_params(ps, 0)
    for epoch in 1:n_epoch
        # p = ProgressMeter.Progress(length(train_loader))

        for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                ŷ = m(x)
                loss(ŷ, y) - logprior(ps, α)
            end
            # for p in ps
                # p .-= Flux.Optimise.apply!(opt, p, gs[p])
            # end
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
