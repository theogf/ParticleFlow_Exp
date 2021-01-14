using DataFrames
using BSON
using Flux
using Flux: Optimise
using Optim
using MLDataUtils
include(srcdir("train_model.jl"))
include(srcdir("utils", "linear.jl"))
include(srcdir("utils", "tools.jl"))


function run_logistic_regression(exp_p)
    @unpack seed, = exp_p
    Random.seed!(seed)
    AVI.setadbackend(:reversediff)

    ## Load the data
    @unpack dataset, use_gpu = exp_p
    dataset = endswith(dataset, ".csv") ? dataset : dataset * ".csv"
    data = CSV.read(datadir("exp_raw", "logistic", dataset), DataFrame; header=true)
    X = Matrix(data[1:end-1])
    y = Vector(data[end])
    # (X_train, y_train), (X_test, y_test) = load_logistic_data(dataset)
    n, n_dim = size(X)

    @info "Loaded dataset $(dataset) with dimension $(size(X))"
    y_pm = sign.(y .- 0.5)
    device = use_gpu ? gpu : cpu

    ## Load experiment parameters
    @unpack B, mf, n_particles, n_iters, k, natmu, alpha, σ_init = exp_p
    @unpack opt_det, opt_stoch, eta, comp_hess = exp_p
    # default values for running experiments

    mf_vals = if mf == :full
        mf = Inf
    elseif mf == :partial
        if dataset == "swarm_flocking"
            mf = vcat(0, 1, 13:12:n_dim)
        else
            error("Partial MF not available for this dataset")
        end
    elseif mf == :none
        false
    end

    logprior(θ) = -0.5 * sum(abs2, θ) / alpha^2
    hists = Vector{Dict}(undef, k)

    for (i, ((X_train, y_train), (X_test, y_test))) in enumerate(kfolds((X, y), obsdim=1, k=k))
        @info "Run $i/$(k)"

        ## Create the model
        function logπ_logistic_stoch(dummy = nothing; B=B, neg=false)
            s = StatsBase.sample(1:n_train, B, replace = false)
            x = X_train[s, :] |> device
            y = y_train[s] |> device
            return function logπ_logistic_stoch(θ)
                loglikelihood = -Flux.Losses.logitbinarycrossentropy(x * θ, y; agg=sum) * n_train / B
                if neg
                    return -(logprior(θ) + loglikelihood)
                else
                    return logprior(θ) + loglikelihood
                end
            end
        end
        function logπ_logistic(θ)
            loglikelihood = -Flux.Losses.logitbinarycrossentropy(X_train * θ, y_train; agg=sum)
            return -(logprior(θ) + loglikelihood)
        end

        n_train = size(X_train, 1)
        # Find the map
        MAP_opt = ADAGrad(0.1)
        MAP_n_iters = 1000
        θ = randn(n_dim) |> device
        @info "Finding map via SGD"
        @info "Init loss : $(logπ_logistic(θ))"
        for i in 1:MAP_n_iters
            g = Flux.gradient(logπ_logistic, θ)
            Flux.update!(MAP_opt, θ, first(g))
            if mod(i, 50) == 0
                @info "i=$i, loss : $(logπ_logistic(θ))"
            end
        end
        @info "Final loss : $(logπ_logistic(θ))"
        
        μ_init = θ #Optim.minimizer(opt_MAP)
        Σ_init = Matrix(σ_init^2 * I(n_dim))
        p_init = MvNormal(μ_init, Σ_init)
        x_init = rand(p_init, n_particles)
        prefix = datadir("results", "logistic", dataset, savename(exp_p))
        ## Create dictionnaries of parameters
        hps = B <= 0 ? nothing : []
        general_p =
            Dict(:hyper_params => hps , :hp_optimizer => nothing, :n_dim => n_dim, :gpu => use_gpu)
        params = Dict{Symbol, Dict}()
        params[:gpf] = Dict(
            :run => exp_p[:gpf],
            :n_particles => n_particles,
            :max_iters => n_iters,
            :natmu => natmu,
            :opt => opt_det(eta),
            :callback => wrap_heavy_cb(;path=joinpath(prefix, savename("gpf", @dict i))),
            :mf => mf_vals,
            :init => x_init,
            :gpu => use_gpu,
            )
        params[:gf] = Dict(
            :run => exp_p[:gf],
            :n_samples => n_particles,
            :max_iters => n_iters,
            :natmu => natmu,
            :opt => opt_stoch(eta),
            :callback => wrap_heavy_cb(;path=joinpath(prefix, savename("gf", @dict i))),
            :init => meancov_to_gf(μ_init, Σ_init, mf),
            :mf => mf_vals,
        )
        params[:dsvi] = Dict(
            :run => exp_p[:dsvi] && !natmu,
            :n_samples => n_particles,
            :max_iters => n_iters,
            :opt => opt_stoch(eta),
            :callback => wrap_heavy_cb(;path=joinpath(prefix, savename("dsvi", @dict i))),
            :init => meancov_to_dsvi(μ_init, Σ_init, mf),
            :mf => mf_vals,
        )
        params[:fcs] = Dict(
            :run => exp_p[:dsvi] && !natmu,
            :n_samples => n_particles,
            :max_iters => n_iters,
            :opt => opt_stoch(eta),
            :callback => wrap_heavy_cb(;path=joinpath(prefix, savename("fcs", @dict i))),
            :init => meancov_to_fcs(μ_init, Σ_init, mf),
            :mf => mf_vals,
        )
        params[:iblr] = Dict(
            :run => exp_p[:dsvi] && natmu,
            :n_samples => n_particles,
            :max_iters => n_iters,
            :opt => Descent(eta),
            :comp_hess => comp_hess,
            :callback => wrap_heavy_cb(;path=joinpath(prefix, savename("iblr", @dict i))),
            :init => meancov_to_iblr(μ_init, Σ_init, mf),
            :mf => mf_vals,
        )

        # Train all models
        hist, params =
            train_model(logπ_logistic, general_p, params)
        hists[i] = hist
    end

    file_prefix = savename(exp_p)
    for alg in algs
        vals = [hist[alg] for hist in hists]
        save(
            datadir("results", "logistic", dataset, file_prefix * string(alg) * ".bson"),
            merge(exp_p, @dict vals)
        )
    end
end
