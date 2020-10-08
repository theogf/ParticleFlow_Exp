using DrWatson
@quickactivate
include(projectdir("analysis", "post_process.jl"))
include(srcdir("utils", "gp.jl"))
include(srcdir("utils", "tools.jl"))
using AxisArrays, MCMCChains, PDMats, KernelFunctions
using StatsFuns, AugmentedGaussianProcesses
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

mu_mc, sig_mc = predict_f(m_mc, X_test, cov = true); sig_mc = vec(sig_mc)
mu_vi, sig_vi = predict_f(m_vi, X_test, cov = true)
pred_mc, sig_mc = proba_y(m_mc, X_test)
pred_vi, sig_vi = proba_y(m_vi, X_test)
acc_mc = mean((pred_mc .> 0.5) .== y_test)
acc_vi = mean((pred_vi .> 0.5) .== y_test)
nll_mc = Flux.Losses.binarycrossentropy(pred_mc, y_test)
nll_vi = Flux.Losses.binarycrossentropy(pred_vi, y_test)

x_mc = collect(eachrow(dropdims(m_mc.inference.sample_store, dims=3)))
N_mc = length(x_mc); μ_mc = ones(N_mc) / N_mc
d_vi = MvNormal(AGP.mean(m_vi.f[1]), AGP.cov(m_vi.f[1]))
wass_vi = wasserstein_semidiscrete(d_vi, x_mc, μ_mc, 0.1)

## Treating the GPF results
accs, acc, acc_sig = [], [], []
nlls, nll, nll_sig = [], [], []
mu_f, sig_f = [], []
wasss, wass, wass_sig = [], [], []
n_parts = vcat(1:9, 10:10:99, 100:50:300)
for n_particles in n_parts
    # n_particles = 10
    gpf_res = collect_results(datadir("results", "gp", dataset, @savename n_particles))
    pred_gpf, sig_gpf, acc_gpf, nll_gpf, wass_gpf = [], [], [], [], []
    for q in gpf_res.q
        f = pred_f(q.x)
        mu_f, sig_f = StatsBase.mean_and_var(f)
        pred, sig = StatsBase.mean_and_var(x->StatsFuns.logistic.(x), f)
        N_gpf = q.n_particles ; ν_gpf = ones(N_gpf) / N_gpf
        wass_val = wasserstein_discrete(x_mc, μ_mc, collect(eachcol(q.x)), ν_gpf, 0.01; N = 100)
        push!(pred_gpf, pred); push!(sig_gpf, sig)
        push!(acc_gpf, mean((pred .> 0.5) .== y_test))
        push!(nll_gpf, Flux.Losses.binarycrossentropy(pred, y_test))
        push!(wass_gpf, wass_val)
    end
    x,y = mean_and_var(acc_gpf)
    push!(acc, x); push!(acc_sig, y); push!(accs, acc_gpf)
    x,y = mean_and_var(nll_gpf)
    push!(nll, x); push!(nll_sig, y); push!(nlls, nll_gpf)
    x,y = mean_and_var(wass_gpf)
    push!(wass, x); push!(wass_sig, y); push!(wasss, wass_gpf)
end

## Plot accuracy
overwrite = true
p = plot(xaxis = :log)
scatter!.([[x] for x in n_parts[1:length(accs)]], accs, msize = 2.0, markerstrokewidth = 0.0, label="", color = :black, alpha= 0.5)
plot!(n_parts, acc, ribbon=sqrt.(acc_sig), label = "GPF", color = colors[1], xlabel = "# Particles", ylabel = "Accuracy")
hline!([acc_mc], label = "MCMC", color = colors[2])
hline!([acc_vi], label = "VI", line = :dash, color = colors[3])
isdir(plotsdir("gp")) ? nothing : mkpath(plotsdir("gp"))
savefig(plotsdir("gp", "Accuracy.png"))
display(p)
if overwrite
    cp(plotsdir("gp", "Accuracy.png"), joinpath("/home/theo","Tex Projects", "GaussianParticleFlow", "figures", "gp", "Accuracy.png"), force =true)
end

p = plot(xaxis = :log)
plot!(n_parts, nll, ribbon=sqrt.(nll_sig), label = "GPF", color = colors[1], xlabel = "# Particles", ylabel = "Neg. Log-Likelihood")
scatter!.([[x] for x in n_parts[1:length(nlls)]], nlls, msize = 2.0, markerstrokewidth = 0.0, label="", color = :black, alpha= 0.5)
hline!([nll_mc], label = "MCMC", color = colors[2])
hline!([nll_vi], label = "VI", line = :dash, color = colors[3])
isdir(plotsdir("gp")) ? nothing : mkpath(plotsdir("gp"))
savefig(plotsdir("gp", "NLL.png"))
display(p)
if overwrite
    cp(plotsdir("gp", "NLL.png"), joinpath("/home/theo","Tex Projects", "GaussianParticleFlow", "figures", "gp", "NLL.png"), force =true)
end

p = plot(n_parts, wass, ribbon=sqrt.(wass_sig), label = "GPF", color = colors[1], xaxis = :log, xlabel = "# Particles", ylabel = L"W^2")
scatter!.([[x] for x in n_parts[1:length(wasss)]], wasss, msize = 2.0, markerstrokewidth = 0.0, label="", color = :black, alpha= 0.5)
hline!([wass_vi], label = "VI", line = :dash, color = colors[3])
isdir(plotsdir("gp")) ? nothing : mkpath(plotsdir("gp"))
savefig(plotsdir("gp", "Wasserstein.png"))
display(p)
if overwrite
    cp(plotsdir("gp", "Wasserstein.png"), joinpath("/home/theo","Tex Projects", "GaussianParticleFlow", "figures", "gp", "Wasserstein.png"), force= true)
end