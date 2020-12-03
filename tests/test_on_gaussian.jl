using DrWatson
@quickactivate
include(srcdir("vi.jl"))
include(srcdir("utils", "dicts.jl"))
include(srcdir("utils", "optimisers.jl"))
using Distributions, LinearAlgebra, Random
using ProgressMeter
using Flux.Optimise

Random.seed!(42)
D = 2

μ = randn(D)
λ = rand(D)
Q, _ = qr(rand(D, D)) # Create random unitary matrix
# Λ = Diagonal(10.0 .^ range(-1, 2, length = dim))
C = Symmetric(Q * Diagonal(λ) * Q')
# C = XXt(rand(D, D) / sqrt(D))
target = MvNormal(μ, C)
logπ(x) = logpdf(target, x)

C₀ = Matrix(I(D))
μ₀ = zeros(D)
## Run alg
T = 10000
S = 10
NGmu = true
algs = Dict()
algs[:dsvi] = DSVI(copy(μ₀), cholesky(C₀).L, S)
algs[:fcs] = FCS(copy(μ₀), Matrix(sqrt(0.5) * Diagonal(cholesky(C₀).L)), sqrt(0.5) * ones(D), S)
algs[:gpf] = GPF(rand(MvNormal(μ₀, C₀), S), NGmu)
algs[:gf] = GF(copy(μ₀), Matrix(cholesky(C₀).L), S, NGmu)
# algs[:spm] = SPM(copy(μ₀), inv(cholesky(C₀).L), S)
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
opt = Optimiser( Descent(0.01), ClipNorm(1.0))
opt = Optimiser(InverseDecay(), ClipNorm(1.0))
opt = Optimiser(Descent(0.01), InverseDecay())
opt = ADAGrad(0.1)
# opt = ScalarADADelta(0.9)
# opt = Descent(0.01)
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
savefig(plotsdir("Gaussian" * @savename(S, D, NGmu) * ".png"))
## Plot the final status
lim = 3
xrange = range(-lim, lim, length = 200)
yrange = range(-lim, lim, length = 200)
ptruth = contour(xrange, yrange, (x,y)->pdf(target, [x, y]), title = "truth", colorbar=false)
ps = [ptruth]
for (name, alg) in algs
    p = contour(xrange, yrange, (x,y)->pdf(MvNormal(alg), [x, y]), title = acs[name], colorbar=false)
    if alg isa GPF
        scatter!(p, eachrow(alg.X)..., lab="", msw=0.0, alpha = 0.6)
    end
    push!(ps, p)
end
plot(ps...) |> display

## Showing evolution 

q = GPF(rand(MvNormal(μ₀, C₀), S), NGmu)
a = Animation()
opt = ADAGrad(1.0)
@showprogress for i in 1:200
    p = contour(xrange, yrange, (x,y)->pdf(target, [x, y]), title = "i=$i", colorbar=false)
    p = contour!(xrange, yrange, (x,y)->pdf(MvNormal(q), [x, y]), colorbar=false)
    scatter!(p, eachrow(q.X)..., lab="", msw=0.0, alpha = 0.9)
    frame(a)
    update!(q, logπ, opt)
end
gif(a, plotsdir("ADAGrad.gif"), fps = 20)