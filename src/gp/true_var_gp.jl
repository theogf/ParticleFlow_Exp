using AugmentedGaussianProcesses
include(srcdir("train_model.jl"))
include(srcdir("utils", "gp.jl"))
include(srcdir("utils", "tools.jl"))

# function run_true_gp(exp_p)
@unpack seed = exp_p
Random.seed!(seed)

## Loading the model and creating the appropriate function
@unpack dataset = exp_p # Load all variables from the dict exp_p
(X_train, y_train), (X_test, y_test) = load_gp_data(dataset)

ρ = initial_lengthscale(X_train)
k = KernelFunctions.transform(SqExponentialKernel(), 1 / ρ)

## Training true model via Gibbs Sampling
@info "Training model with sampling"
@unpack nSamples, nBurnin = exp_p
# m = MCGP(X_train, y_train, k, LogisticLikelihood(), GibbsSampling(nBurnin = nBurnin))
# chain = AugmentedGaussianProcesses.sample(m, nSamples)
# μ_f, σ_f = predict_f(m, X_test; cov = true, diag = true)
# @show typeof.([μ_f, σ_f])
# mcmc = true; vi = false
# savepath = datadir("results", "gp", dataset)
# tagsave(joinpath(savepath, "mcmcgp.bson"),
#     @dict m mcmc vi;
#     safe=false, storepatch = false)
## Training moodel via variational inference
@info "Training model with VI"
m = VGP(X_train, y_train, k, LogisticLikelihood(), QuadratureVI(optimiser = Descent(0.0001), natural = false), optimiser=nothing, verbose = 3)
m.f[1].post.μ .= AGP.mean(m_mc.f[1])
train!(m, 20000)
pred_vi, sig_vi = proba_y(m, X_test)
nll_vi = Flux.Losses.binarycrossentropy(pred_vi, y_test)
acc_vi = mean((pred_vi .> 0.5) .== y_test)
d_vi = MvNormal(AGP.mean(m.f[1]), AGP.cov(m.f[1]))
wass_vi = wasserstein_semidiscrete(d_vi, x_mc, μ_mc, 0.1)
μ_f, σ_f = predict_f(m, X_test; cov = true, diag = true)
gpvi = true; gpmcmc = false
post = AGP.posterior(m.f[1])
savepath = datadir("results", "gp", dataset)
tagsave(joinpath(savepath, "vigp.bson"),
    @dict post gpvi gpmcmc;
    safe=false, storepatch = false)
# end
