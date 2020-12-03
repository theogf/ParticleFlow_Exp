using DrWatson
@quickactivate
include(srcdir("vi.jl"))
include(srcdir("utils", "dicts.jl"))
using Distributions, LinearAlgebra
using ProgressMeter
using Flux.Optimise
D = 2

μ = randn(D)
C = XXt(rand(D, D) / sqrt(D))
target = MvNormal(μ, C)
logπ(x) = logpdf(target, x)

C₀ = Matrix(I(D))
μ₀ = zeros(D)
## Run alg
T = 2000
S = 3
NGmu = true
algs = Dict()
algs[:dsvi] = DSVI(copy(μ₀), cholesky(C₀).L, S)
algs[:fcs] = FCS(copy(μ₀), Matrix(sqrt(0.5) * Diagonal(cholesky(C₀).L)), sqrt(0.5) * ones(D), S)
algs[:gpf] = GPF(rand(MvNormal(μ₀, C₀), S), NGmu)
algs[:gf] = GF(copy(μ₀), Matrix(cholesky(C₀).L), S, NGmu)
# algs[:ngd] = NGD(copy(μ₀), cholesky(C₀).L)


err_mean = Dict()
err_cov = Dict()
times = Dict()
for (name, alg) in algs
    err_mean[name] = zeros(T+1)
    err_mean[name][1] = norm(mean(alg) - μ)
    err_cov[name] = zeros(T+1)
    err_cov[name][1] = norm(cov(alg) - C)
    times[name] = 0
end


opt = Optimiser(ADAM(1.0), IncreasingRate(0.1, 1e-4))
opt = Optimiser(LogLinearIncreasingRate(0.1, 1e-6, 100), ClipNorm(sqrt(D)))
opt = Optimiser(ClipNorm(D), Descent(0.001))
opt = Descent(0.01)
# opt = ADAM(0.1)
@showprogress for i in 1:T
    for (name, alg) in algs
        t = @elapsed update!(alg, logπ, opt)
        times[name] += t
        err_mean[name][i+1] = norm(mean(alg) - μ)
        err_cov[name][i+1] = norm(cov(alg) - C)
    end
end
for (name, alg) in algs
    @info "$name :\nDiff mean = $(norm(mean(alg) - μ))\nDiff cov = $(norm(cov(alg) - C))\nTime : $(times[name])"
end

gpf = algs[:gpf]
gf = algs[:gf]

# Plotting difference
using Plots
p_m = plot(title = "Mean error", yaxis=:log)
p_C = plot(title = "Cov error", yaxis=:log)
for (name, alg) in algs
    cut = findfirst(x->x==0, err_mean[name])
    cut = isnothing(cut) ? T : cut
    plot!(p_m, 1:cut, err_mean[name][1:cut], lab = acs[name])
    plot!(p_C, 1:cut, err_cov[name][1:cut], lab = acs[name])
end

plot(p_m, p_C) |> display
savefig(plotsdir("Gaussian" * @savename(S,D) * ".png"))
## Plot the final status