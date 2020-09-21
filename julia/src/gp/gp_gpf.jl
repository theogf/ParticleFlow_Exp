include(srcdir("train_model.jl"))
include(srcdir("utils", "tools.jl"))
include(srcdir("utils", "gp.jl"))

using PDMats
using Optim
using StatsFuns: logistic
function run_gp_gpf(exp_p)
    @unpack seed = exp_p
    Random.seed!(seed)
    AVI.setadbackend(:reversediff)

    @unpack dataset = exp_p # Load all variables from the dict exp_p
    (X_train, y_train), (X_test, y_test) = load_gp_data(dataset)
    y_trainpm = sign.(y_train .- 0.5)
    n_train, n_dim = size(X_train)
    ρ = initial_lengthscale(X_train)
    k = transform(SqExponentialKernel(), ρ)

    K = kernelpdmat(k, X_train; obsdim = 1)
    prior = TuringDenseMvNormal(zeros(n_train), K)
    # Define the log joint function
    function logπ(θ)
        -Flux.Losses.logitbinarycrossentropy(θ, y_train; agg = sum) + logpdf(prior, θ)
    end
    # And its negative counterpart for Optim
    neglogπ(θ) = -logπ(θ)
    # The gradient can be given in closed form
    function gradneglogπ!(G, θ)
        G .= -y_trainpm .* logistic.(-y_trainpm .* θ) + K \ θ
    end

    # Optimize the MAP as a starting point via LBFGS
    opt_MAP = optimize(
        neglogπ,
        gradneglogπ!,
        Vector(y_trainpm),
        LBFGS(),
        Optim.Options(allow_f_increases = false, iterations = 10),
    )
    @show opt_MAP

    # Unpack necessary parameters to the experiment
    @unpack n_particles, n_iters, n_runs, cond1, cond2, σ_init = exp_p

    @unpack opt = exp_p


    for i = 1:n_runs
        file_prefix = @savename n_particles σ_init n_iters n_runs i
        @info "Run $i/$(n_runs)"
        ## Create dictionnaries of parameters
        x_init = rand(MvNormal(Optim.minimizer(opt_MAP), σ_init), n_particles)
        general_p =
            Dict(:hyper_params => nothing, :hp_optimizer => nothing, :n_dim => n_dim, :gpu => false)
        gflow_p = Dict(
            :run => true,
            :n_particles => n_particles,
            :max_iters => n_iters,
            :cond1 => cond1,
            :cond2 => cond2,
            :opt => deepcopy(opt),
            :callback => nothing,
            :mf => false,
            :init => x_init,
        )
        no_run = Dict(:run => false)
        vi, q = init_gflow(gflow_p, general_p)
        device = general_p[:gpu] ? gpu : cpu
        AVI.vi(
            logπ,
            vi,
            q |> device,
            optimizer = gflow_p[:opt] |> device,
            hyperparams = nothing,
            hp_optimizer = nothing,
            callback = nothing,
        )
        tagsave(
        datadir("results", "gp", dataset, file_prefix * ".bson"),
        @dict n_particles σ_init n_iters n_runs cond1 cond2 exp_p d_target i;
        safe = false,
        storepatch = false,
        )
    end
end
