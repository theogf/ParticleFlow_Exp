using DataFrames
using BSON
using Flux
using CUDA
using Optim
include(srcdir("train_model.jl"))
include(srcdir("utils", "tools.jl"))
include(srcdir("utils", "linear.jl"))
(X_train, y_train), (X_test, y_test) = load_logistic_data("swarm_flocking")

function cb_val(h, i, q, hp)

end

function run_logistic_regression(exp_p)
    @unpack seed = exp_p
    Random.seed!(seed)
    AVI.setadbackend(:reversediff)

    ## Load the data
    @unpack dataset, use_gpu = exp_p
    # (X_train, y_train), (X_test, y_test) = load_logistic_data(dataset)
    n_train, n_dim = size(X_train)
    y_trainpm = sign.(y_train .- 0.5)
    device = use_gpu ? gpu : cpu
    B = 200

    ## Load experiment parameters
    @unpack n_particles, n_iters, n_runs, cond1, cond2, α, σ_init = exp_p


    mf = vcat(0, 1, 13:12:n_dim)
    Flux.@functor TuringDiagMvNormal
    prior = TuringDiagMvNormal(zeros(n_dim), α * ones(n_dim)) |> device
    @show typeof(prior)
    ## Create the model
    function logπ_logistic(dummy = nothing;B = B, neg = false)
        s = StatsBase.sample(1:n_train, B, replace = false)
        x = X_train[s, :] |> device
        y = y_train[s] |> device
        return function logπ_logistic_stoch(θ)
            logprior = logpdf(prior, θ)
            loglikelihood = -Flux.Losses.logitbinarycrossentropy(x * θ, y; agg=sum) * n_train / B
            if neg
                return -(logprior + loglikelihood)
            else
                return logprior + loglikelihood
            end
        end
    end
    # neglogπ(θ) = -logπ_logistic(θ)
    # The gradient can be given in closed form
    # function gradneglogπ!(G, θ)
        # G .= -y_trainpm .* logistic.(-y_trainpm .* (X_train * θ)) + θ ./ α
    # end
    MAP_opt = ADAGrad(0.1)
    MAP_n_iters = 500
    θ = randn(n_dim) |> device
    @info "Init loss : $(logπ_logistic(B=n_train)(θ))"
    for i in 1:MAP_n_iters
        g = Flux.gradient(logπ_logistic(B; neg=true), θ)
        Flux.update!(MAP_opt, θ, first(g))
        if mod(i, 50) == 0
            @info "i=$i, loss : $(logπ_logistic(B=n_train)(θ))"
        end
    end
    @info "Final loss : $(logπ_logistic(B=n_train)(θ))"

    # opt_MAP = optimize(
    #     neglogπ,
    #     # gradneglogπ!,
    #     randn(n_dim),
    #     LBFGS(),
    #     Optim.Options(iterations = 50)
    # )
    # @show opt_MAP

    gpf = []
    advi = []
    steinvi = []

    for i in 1:n_runs
        @info "Run $i/$(n_runs)"
        opt = exp_p[:opt]
        μ_init = θ #Optim.minimizer(opt_MAP)
        Σ_init = σ_init^2 * I(n_dim)
        p_init = MvNormal(μ_init, Σ_init)
        x_init = rand(p_init, n_particles)

        ## Create dictionnaries of parameters
        general_p =
            Dict(:hyper_params => [] , :hp_optimizer => nothing, :n_dim => n_dim, :gpu => use_gpu)
        gflow_p = Dict(
            :run => exp_p[:gpf],
            :n_particles => n_particles,
            :max_iters => n_iters,
            :cond1 => cond1,
            :cond2 => cond2,
            :opt => deepcopy(opt),
            :callback => no_cb,
            :mf => mf,
            :init => x_init,
            :gpu => use_gpu,
        )
        advi_p = Dict(
            :run => exp_p[:advi] && !cond1 && !cond2,
            :n_samples => n_particles,
            :max_iters => n_iters,
            :opt => deepcopy(opt),
            :callback => no_cb,
            :init => (μ_init, sqrt.(Σ_init)),
            :mf => mf,
        )
        stein_p = Dict(
            :run => exp_p[:steinvi] && !cond1 && !cond2,
            :n_particles => n_particles,
            :max_iters => n_iters,
            :kernel => KernelFunctions.transform(SqExponentialKernel(), 1.0),
            :opt => deepcopy(opt),
            :callback => no_cb,
            :init => x_init,
        )

        # Train all models
        _gpf, _advi, _steinvi =
            train_model(logπ_logistic, general_p, gflow_p, advi_p, stein_p)
        push!(gpf, _gpf)
        push!(advi, _advi)
        push!(steinvi, _steinvi)
    end

    file_prefix = savename(exp_p)

    tagsave(datadir("results", "linear", file_prefix * ".bson"),
            merge(exp_p, @dict gpf advi steinvi);
            safe=false, storepatch = false)
end
