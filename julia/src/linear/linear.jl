using DataFrames
using BSON
using Flux
using Optim
include(srcdir("train_model.jl"))
include(srcdir("utils", "tools.jl"))
function run_logistic_regression(exp_p)
    @unpack seed = exp_p
    Random.seed!(seed)
    AVI.setadbackend(:reversediff)

    ## Load the data
    @unpack dataset = exp_p
    (X_train, y_train), (X_test, y_test) = get_logistic_data(dataset)
    n_train, n_dim = size(X_train)

    ## Load experiment parameters
    @unpack n_particles, n_iters, n_runs, cond1, cond2, α, σ_init = exp_p

    prior = TuringDiagMvNormal(zeros(n_dim), α)
    ## Create the model
    function logπ_logistic(θ)
        logprior = logpdf(prior, θ)
        loglikelihood = -Flux.logitbinarycrossentropy(X * θ, y_train)
        return logprior + loglikelihood
    end
    neglogπ(θ) = -logπ_logistic(θ)
    # The gradient can be given in closed form
    function gradneglogπ!(G, θ)
        G .= -y_trainpm .* logistic.(-y_trainpm .* θ) + θ ./ α
    end

    opt_MAP = optimize(
        neglogπ,
        gradneglogπ!,
        randn(n_dim),
        LBFGS(),
        Optim.Options(iteraitons = 50)
    )
    @show opt_MAP

    gpf = []
    advi = []
    steinvi = []

    for i in 1:n_runs
        @info "Run $i/$(n_runs)"
        opt = exp_p[:opt]
        μ_init = Optim.minimizer(opt_MAP)
        Σ_init = σ_init^2 * I(n_dim)
        p_init = MvNormal(μ_init, Σ_init)
        x_init = rand(p_init, n_particles)

        ## Create dictionnaries of parameters
        general_p =
            Dict(:hyper_params => nothing, :hp_optimizer => ADAGrad(0.1), :n_dim => dim, :gpu => false)
        gflow_p = Dict(
            :run => exp_p[:gpf],
            :n_particles => n_particles,
            :max_iters => n_iters,
            :cond1 => cond1,
            :cond2 => cond2,
            :opt => deepcopy(opt),
            :callback => wrap_cb(),
            :mf => false,
            :init => x_init,
        )
        advi_p = Dict(
            :run => exp_p[:advi] && !cond1 && !cond2,
            :n_samples => n_particles,
            :max_iters => n_iters,
            :opt => deepcopy(opt),
            :callback => wrap_cb(),
            :init => (μ_init, sqrt.(Σ_init)),
        )
        stein_p = Dict(
            :run => exp_p[:steinvi] && !cond1 && !cond2,
            :n_particles => n_particles,
            :max_iters => n_iters,
            :kernel => KernelFunctions.transform(SqExponentialKernel(), 1.0),
            :opt => deepcopy(opt),
            :callback => wrap_cb(),
            :init => x_init,
        )

        # Train all models
        _gpf, _advi, _steinvi =
            train_model(logπ_gauss, general_p, gflow_p, advi_p, stein_p)
        push!(gpf, _gpf)
        push!(advi, _advi)
        push!(steinvi, _steinvi)
    end

    file_prefix = save_name(ps[1])

    tagsave(datadir("results", "gaussian", file_prefix * ".bson"),
            @dict dim n_particles full_cov n_iters n_runs cond1 cond2 gpf advi steinvi exp_p d_target;
            safe=false, storepatch = false)
end
