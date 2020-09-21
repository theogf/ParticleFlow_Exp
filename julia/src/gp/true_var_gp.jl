include(srcdir("train_model.jl"))
include(srcdir("utils", "gp.jl"))
include(srcdir("utils", "tools.jl"))
function run_true_gp(exp_p)
    @unpack seed = exp_p
    Random.seed!(seed)

    ## Loading the model and creating the appropriate function
    @unpack dataset = exp_p # Load all variables from the dict exp_p
    (X_train, y_train), (X_test, y_test) = load_gp_data(dataset)

    ρ = initial_lengthscale(X_train)
    k = transform(SqExponentialKernel(), ρ)

    ## Training true model via Gibbs Sampling
    @info "Training model with sampling"
    m = MCGP(X_train, y_train, k, LogisticLikelihood(), GibbSampling(burnin = 200))
    chain = sample(m, 10_000)
    μ_f, σ_f = predict_f(m, X_train; cov = true, diag = true)
    mcmc = true; vi = false; gpf = false
    savepath = datadir("results", "gp", dataset)
    tagsave(joinpath(savepath, "mcmcgp.bson"),
        @dict chain mcmc vi gpf y_test μ_f, σ_f;
        safe=false, storepatch = false)
    ## Training moodel via variational inference
    @info "Training model with VI"
    m = VGP(X_train, y_train, k, LogisticLikelihood(), AnalyticVI(), optimiser=nothing)
    train!(m; iterations = 20)
    μ_f, σ_f = predict_f(m, X_train; cov = true, diag = true)
    vi = true; mcmc = false; gpf = false
    savepath = datadir("results", "gp", dataset)
    tagsave(joinpath(savepath, "vigp.bson"),
        @dict c vi mcmc vi gpf y_test μ_f σ_f;
        safe=false, storepatch = false)
end
