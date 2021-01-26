using DataFrames
using BSON
using ForwardDiff
using Flux
using Flux: Optimise
using Optim
using MLDataUtils
using CSV
using StatsFuns: logistic
include(srcdir("train_model.jl"))
# include(srcdir("utils", "linear.jl"))
include(srcdir("utils", "tools.jl"))

function run_logistic_regression(exp_p)
    @unpack seed, = exp_p
    Random.seed!(seed)
    AVI.setadbackend(:reversediff)

    ## Load the data
    @unpack dataset = exp_p
    dataset_file = endswith(dataset, ".csv") ? dataset : dataset * ".csv"
    data = CSV.read(datadir("exp_raw", "logistic", dataset_file), DataFrame; header=true)
    X = Matrix(data[1:end-1])
    rescale!(X, obsdim=1)
    y = Vector(data[end])
    # (X_train, y_train), (X_test, y_test) = load_logistic_data(dataset)
    n, n_dim = size(X)

    @info "Loaded dataset $(dataset) with dimension $(size(X))"
    y_pm = sign.(y .- 0.5)
    device = cpu

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
        n_train = size(X_train, 1)

        ## Create the model
        function logπ_logistic_stoch(dummy = nothing; B=B)
            s = StatsBase.sample(1:n_train, B, replace = false)
            x = X_train[s, :] |> device
            y = y_train[s] |> device
            return function logπ_logistic_stoch(θ)
                loglikelihood = -Flux.Losses.logitbinarycrossentropy(x * θ, y; agg=sum) * n_train / B
                return logprior(θ) + loglikelihood
            end
        end
        function logπ_logistic(θ)
            loglikelihood = -Flux.Losses.logitbinarycrossentropy(X_train * θ, y_train; agg=sum)
            return logprior(θ) + loglikelihood
        end

        # Find the map
        MAP_opt = ADAGrad(0.1)
        MAP_n_iters = 1000
        θ = randn(n_dim) |> device
        @info "Finding map via SGD"
        @info "Init loss : $(-logπ_logistic(θ))"
        for i in 1:MAP_n_iters
            g = Flux.gradient(logπ_logistic, θ)
            Flux.update!(MAP_opt, θ, -first(g))
            if mod(i, 50) == 0
                @info "i=$i, loss : $(-logπ_logistic(θ))"
            end
        end
        @info "Final loss : $(-logπ_logistic(θ))"
        
        function cb_val(h, i, q, hp)
            pred_X_train = pred_logistic(q, X_train)
            pred_X_test = pred_logistic(q, X_test)
            push!(h, :acc_train, i, accuracy(pred_X_train, y_train))
            push!(h, :acc_test, i, accuracy(pred_X_test, y_test))
            push!(h, :nll_train, i, nll_logistic(pred_X_train, y_train))
            push!(h, :nll_test, i, nll_logistic(pred_X_test, y_test))
            # push!(h, :elbo, i, elbo(q))
        end

        μ_init = θ #Optim.minimizer(opt_MAP)
        μ_init = randn(n_dim)
        Σ_init = Matrix(σ_init^2 * I(n_dim))
        # Σ_init = Matrix(Symmetric(-inv(ForwardDiff.hessian(logπ_logistic, θ)))) 
        p_init = MvNormal(μ_init, Σ_init)
        x_init = rand(p_init, n_particles)
        prefix = datadir("results", "logistic", dataset, savename(exp_p))
        ## Create dictionnaries of parameters
        hps = B <= 0 ? nothing : []
        general_p =
            Dict(:hyper_params => hps , :hp_optimizer => nothing, :n_dim => n_dim, :gpu => false)
        params = Dict{Symbol, Dict}()
        params[:gpf] = Dict(
            :run => exp_p[:gpf],
            :n_particles => n_particles,
            :max_iters => n_iters,
            :natmu => natmu,
            :opt => (@eval $opt_det($eta)),
            :callback => wrap_cb(;cb_val=cb_val),
            # :callback => wrap_heavy_cb(;path=joinpath(prefix, savename("gpf", @dict i))),
            :mf => mf_vals,
            :init => copy(x_init),
            :gpu => false,
        )
        params[:gf] = Dict(
            :run => exp_p[:gf],
            :n_samples => n_particles,
            :max_iters => n_iters,
            :natmu => natmu,
            :opt => (@eval $opt_stoch($eta)),
            :callback => wrap_cb(;cb_val=cb_val),
            # :callback => wrap_heavy_cb(;path=joinpath(prefix, savename("gf", @dict i))),
            :init => meancov_to_gf(μ_init, Σ_init, mf),
            :mf => mf_vals,
        )
        params[:dsvi] = Dict(
            :run => exp_p[:dsvi] && !natmu,
            :n_samples => n_particles,
            :max_iters => n_iters,
            :opt => (@eval $opt_stoch($eta)),
            :callback => wrap_cb(;cb_val=cb_val),
            # :callback => wrap_heavy_cb(;path=joinpath(prefix, savename("dsvi", @dict i))),
            :init => meancov_to_dsvi(μ_init, Σ_init, mf),
            :mf => mf_vals,
        )
        params[:fcs] = Dict(
            :run => exp_p[:fcs] && !natmu && mf == :none,
            :n_samples => n_particles,
            :max_iters => n_iters,
            :opt => (@eval $opt_stoch($eta)),
            :callback => wrap_cb(;cb_val=cb_val),
            # :callback => wrap_heavy_cb(;path=joinpath(prefix, savename("fcs", @dict i))),
            :init => meancov_to_fcs(μ_init, Σ_init, mf),
            :mf => mf_vals,
        )
        params[:iblr] = Dict(
            :run => exp_p[:iblr],
            :n_samples => n_particles,
            :max_iters => n_iters,
            :opt => Descent(eta),
            :comp_hess => comp_hess,
            :callback => wrap_cb(;cb_val=cb_val),
            # :callback => wrap_heavy_cb(;path=joinpath(prefix, savename("iblr", @dict i))),
            :init => meancov_to_iblr(μ_init, Σ_init, mf),
            :mf => mf_vals,
        )
        params[:svgd] = Dict(
            :run => exp_p[:svgd],
            :n_particles => n_particles,
            :max_iters => n_iters,
            :opt => (@eval $opt_det($eta)),
            :callback => wrap_cb(;cb_val=cb_val),
            # :callback => wrap_heavy_cb(;path=joinpath(prefix, savename("gpf", @dict i))),
            :mf => mf_vals,
            :init => copy(x_init),
            :gpu => false,
        )

        # Train all models
        hist, params =
            train_model(B <= 0 ? logπ_logistic : logπ_logistic_stoch, general_p, params)
        hists[i] = hist
    end
    for alg in algs
        delete!(exp_p, alg)
    end
    file_prefix = savename(exp_p)
    for alg in algs
        vals = [hist[alg] for hist in hists]
        save(
            datadir("results", "logistic", dataset, file_prefix * "_" * string(alg) * ".bson"),
            merge(exp_p, @dict vals)
        )
    end
end

function pred_logistic(q, X; nsamples=1000)
    mean(logistic.(X * w) for w in eachcol(rand(q, nsamples)))
end

function accuracy(p_y_pred, y_true)
    mean((p_y_pred .> 0.5) .== y_true)
end

function nll_logistic(p_y_pred, y_true)
    mean(logpdf.(Bernoulli.(p_y_pred), y_true))
end