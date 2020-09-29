using DrWatson
@quickactivate
include(projectdir("analysis", "post_process.jl"))
include(srcdir("utils", "gp.jl"))
include(srcdir("utils", "tools.jl"))
using AxisArrays, MCMCChains, PDMats, KernelFunctions
using StatsFuns
dataset = "ionosphere"
(X_train, y_train), (X_test, y_test) = load_gp_data(dataset)

ρ = initial_lengthscale(X_train)
k = KernelFunctions.transform(SqExponentialKernel(), 1 / ρ)
K = kernelpdmat(k, X_train, obsdim = 1)
Ktt = kerneldiagmatrix(k, X_test, obsdim = 1)
Ktx = kernelmatrix(k, X_test, X_train, obsdim = 1)
function pred_f(f)
    Ktx * inv(K) * f
end

base_res = collect_results(datadir("results", "gp", dataset))
vi_res = @linq base_res |> where(:vi .=== true)
mc_res = @linq base_res |> where(:mcmc .=== true)
m_vi = vi_res.m[1]
m_mc = mc_res.m[1]

pred_mc, sig_mc = proba_y(m_mc, X_test)
pred_vi, sig_vi = proba_y(m_vi, X_test)
acc_mc = mean((pred_mc .> 0.5) .== y_test)
acc_vi = mean((pred_vi .> 0.5) .== y_test)
nll_mc = Flux.Losses.binarycrossentropy(pred_mc, y_test)
nll_vi = Flux.Losses.binarycrossentropy(pred_vi, y_test)


n_particles = 10
gpf_res = collect_results(datadir("results", "gp", dataset, @savename n_particles))
pred_gpf, sig_gpf, acc_gpf, nll_gpf = [], [], [], []
for q in gpf_res.q
    f = pred_f(q.x)
    pred, sig = StatsBase.mean_and_var(x->StatsFuns.logistic.(x), f)
    push!(pred_gpf, pred); push!(sig_gpf, sig)
    push!(acc_gpf, mean((pred .> 0.5) .== y_test))
    push!(nll_gpf, Flux.Losses.binarycrossentropy(pred, y_test))
end

acc_gpf[1]
nll_gpf[1]
