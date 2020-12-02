using DrWatson
@quickactivate
include(srcdir("vi.jl"))
# const VI = VariationalInference
using Distributions, LinearAlgebra
using ProgressMeter
using Flux.Optimise
D = 2

μ = [3.0, 4.0]
C = XXt(rand(D, D))
target = MvNormal(μ, C)
logπ(x) = logpdf(target, x)

C₀ = Matrix(I(D))
μ₀ = zeros(D)
## Run alg
algs = Dict()
algs[:dsvi] = DSVI(copy(μ₀), cholesky(C₀).L)
algs[:fcs] = FCS(copy(μ₀), Matrix(sqrt(0.5) * Diagonal(cholesky(C₀).L)), sqrt(0.5) * ones(D))
algs[:ngd] = NGD(copy(μ₀), cholesky(C₀).L)

T = 2000

err_mean = Dict()
err_cov = Dict()
for (name, alg) in algs
    err_mean[name] = zeros(T)
    err_cov[name] = zeros(T)
end


opt = Optimiser(ADAM(1.0), IncreasingRate(0.1, 1e-4))
opt = LogLinearIncreasingRate(0.1, 1e-6)
@showprogress for i in 1:T
    for (name, alg) in algs
        update!(alg, logπ, opt)
        err_mean[name][i] = norm(mean(alg) - μ)
        err_cov[name][i] = norm(cov(alg) - C)
    end
end
for (name, alg) in algs
    @info "$name :\nDiff mean = $(norm(mean(alg) - μ))\nDiff cov = $(norm(cov(alg) - C))"
end

## Plotting difference
using Plots
p_m = plot(title = "Mean error")
p_C = plot(title = "Cov error")
for (name, alg) in algs
    cut = findfirst(x->x==0, err_mean[name])
    cut = isnothing(cut) ? T : cut
    plot!(p_m, 1:cut, err_mean[name][1:cut], lab = string(name))
    plot!(p_C, 1:cut, err_cov[name][1:cut], lab = string(name))
end
plot(p_m, p_C)