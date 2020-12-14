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

logbanana(x1, x2) = -0.5(0.01 * x1^2 + 0.1(x2 + 0.1x1^2 - 10)^2)
logbanana(x) = logbanana(first(x), last(x))
banana(x1, x2) = exp(logbanana(x1, x2))
banana(x) = exp(logbanana(x))
logπ(x) = logbanana(x)

C₀ = Matrix(I(D))
μ₀ = zeros(D)
## Run alg
T = 1000
S = 100
Stest = 1000
NGmu = true
algs = Dict()
algs[:dsvi] = DSVI(copy(μ₀), cholesky(C₀).L, S)
algs[:fcs] = FCS(copy(μ₀), Matrix(sqrt(0.5) * Diagonal(cholesky(C₀).L)), sqrt(0.5) * ones(D), S)
algs[:gpf] = GPF(rand(MvNormal(μ₀, C₀), S), NGmu)
algs[:gf] = GF(copy(μ₀), Matrix(cholesky(C₀).L), S, NGmu)
# algs[:iblr] = IBLR(copy(μ₀), inv(C₀), S)

# algs[:spm] = SPM(copy(μ₀), inv(cholesky(C₀).L), S)
# algs[:ngd] = NGD(copy(μ₀), cholesky(C₀).L)


ELBOs = Dict()
# err_cov = Dict()
times = Dict()
for (name, alg) in algs
    ELBOs[name] = zeros(T+1)
    ELBOs[name][1] = ELBO(alg, logπ, nSamples = Stest)
    times[name] = 0
end


opt = Optimiser(ADAM(1.0), IncreasingRate(0.1, 1e-4))
opt = Optimiser(LogLinearIncreasingRate(0.1, 1e-6, 100), ClipNorm(sqrt(D)))
opt = Optimiser( Descent(0.01), ClipNorm(1.0))
opt = Optimiser(InverseDecay(), ClipNorm(1.0))
opt = Optimiser(Descent(0.01), InverseDecay())
# opt = Momentum(0.001)
# opt = ScalarADADelta(0.9)
# opt = Descent(0.01)
# opt = ADAM(0.1)
opt = MatADAGrad(0.1)
@showprogress for i in 1:T
    for (name, alg) in algs
        t = @elapsed update!(alg, logπ, opt)
        times[name] += t
        ELBOs[name][i+1] = ELBO(alg, logπ, nSamples = Stest)
    end
end
for (name, alg) in algs
    @info "$name :\nELBO = $(ELBO(alg, logπ, nSamples = Stest))\nTime : $(times[name])"
end

# Plotting difference
using Plots
p_L = plot(title = "ELBO")
for (name, alg) in algs
    cut = findfirst(x->x==0, ELBOs[name])
    cut = isnothing(cut) ? T : cut
    plot!(p_L, 1:cut, ELBOs[name][1:cut], lab = acs[name])
end

p_L |> display
savefig(plotsdir("Banana" * @savename(S, NGmu) * ".png"))
## Plot the final status
lim = 20
xrange = range(-lim, lim, length = 200)
yrange = range(-lim, lim, length = 200)
ptruth = contour(xrange, yrange, banana, title = "truth", colorbar=false)
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
opt = Momentum(0.1)
opt = Descent(1.0)
opt = MatADAGrad(1.0)
@showprogress for i in 1:200
    if i % 10 == 0
        p = contour(xrange, yrange, logbanana, title = "i=$i", colorbar=false)
        contour!(p, xrange, yrange, (x,y)->logpdf(MvNormal(q), [x, y]), colorbar=false)
        scatter!(p, eachrow(q.X)..., lab="", msw=0.0, alpha = 0.9)
        frame(a)
    end
    update!(q, logπ, opt)
end
gif(a, plotsdir("Banana - Momentum.gif"), fps = 10)