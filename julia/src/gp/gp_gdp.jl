include(srcdir("train_model.jl"))
include(srcdir("utils", "tools.jl"))
include(srcdir("utils", "gp.jl"))

using PDMats # We allow to construct a PDMat matrix
using Optim # For finding the MAP
using StatsFuns: logistic
function run_gp_gdp(exp_p)
    @unpack seed = exp_p
    Random.seed!(seed) # Fix the ranom seed
    AVI.setadbackend(:reversediff)

    @unpack dataset = exp_p # Load all variables from the dict exp_p
    (X_train, y_train), (X_test, y_test) = load_gp_data(dataset) # Loading data as train and test
    y_trainpm = sign.(y_train .- 0.5) # [-1, 1] version of y
    n_train, n_dim = size(X_train) # Dimension of the data
    ρ = initial_lengthscale(X_train) # Initialize kernel lengthscale via median technique
    k = KernelFunctions.transform(SqExponentialKernel(), 1 / ρ)

    K = kernelpdmat(k, X_train; obsdim = 1) # Create the kernel matrix
    Kxtestxtrain = kernelmatrix(k, X_test, X_train; obsdim = 1)
    Kxtestxtest = kerneldiagmatrix(k, X_test; obsdim = 1)
    prior = TuringDenseMvNormal(zeros(n_train), K) # GP Prior
    # Define the log joint function
    function logπ(θ)
        -Flux.Losses.logitbinarycrossentropy(θ, y_train; agg = sum) + logpdf(prior, θ) # Logjoint (we use Flux formulation)
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
        Optim.Options(allow_f_increases = false, iterations = 200),
    )
    @show opt_MAP # Show the final result

    # Unpack necessary parameters to the experiment
    @unpack n_particles, n_iters, n_runs, cond1, cond2, σ_init, opt = exp_p

    for i = 1:n_runs
        file_prefix = savename(merge(exp_p, @dict i)) # Create a prefix for each run
        @info "Run $i/$(n_runs)"
        ## Create dictionnaries of parameters
        μ_init = Optim.minimizer(opt_MAP)
        Γ_init = σ_init * Matrix{Float64}(I(n_train)) # Initialize particle around the MAP
        h = MVHistory()
        general_p =
            Dict(:hyper_params => nothing, :hp_optimizer => nothing, :n_dim => n_dim, :gpu => false)
        gflow_p = Dict(
            :run => true,
            :n_samples => n_particles,
            :max_iters => n_iters,
            :cond1 => cond1,
            :cond2 => cond2,
            :opt => deepcopy(opt),
            :callback => wrap_heavy_cb(; path=datadir("results", "gp", dataset, @savename(n_particles), file_prefix))(h),
            :mf => false,
            :init => (μ_init, Γ_init),
        )
        # We save the state of the training every log10 iterations in a split folder for later debugging
        vi, q = init_gaussflow(gflow_p, general_p) # Initialization of the objects
        device = general_p[:gpu] ? gpu : cpu # Choice of the device
        AVI.vi(
            logπ,
            vi,
            q |> device,
            optimizer = gflow_p[:opt] |> device,
            hyperparams = nothing,
            hp_optimizer = nothing,
            callback = gflow_p[:callback],
        )
        tagsave(
        datadir("results", "gp", dataset, savename("gdp_", @dict(n_particles)), file_prefix * ".bson"),
        merge(Dict(exp_p), @dict k K Kxtestxtest Kxtestxtrain y_train y_test q i h);
        safe = false,
        storepatch = false,
        ) # Saving the final state at the end
    end
end

#
