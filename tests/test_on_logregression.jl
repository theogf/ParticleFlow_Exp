using DrWatson
@quickactivate
include(srcdir("vi.jl"))
include(srcdir("utils", "dicts.jl"))
include(srcdir("utils", "optimisers.jl"))
using Distributions, LinearAlgebra, Random
using ProgressMeter
using Flux.Optimise
using StatsFuns: logistic

Random.seed!(42)
D = 2
N = 30
μs = [[-5, 1], [1, 5]]
stds = [1^2, 1.1^2]
s = shuffle(1:2N)
y = vcat(zeros(N), ones(N))[s]
x = hcat(rand.(MvNormal.(μs, stds), N)...)[:, s]

prior = MvNormal(zeros(2), 10^2)
likelihood(θ::Real) = Bernoulli(logistic(θ))
likelihood(θ::AbstractVector) = Product(likelihood.(θ))
logπ(θ) = logpdf(likelihood(x' * θ), y) + logpdf(prior, θ)

C₀ = Matrix(I(D))
μ₀ = zeros(D)
## Run alg
T = 1000
S = 100
Stest = 100
NGmu = true
algs = Dict()
algs[:dsvi] = DSVI(copy(μ₀), cholesky(C₀).L, S)
algs[:fcs] = FCS(copy(μ₀), Matrix(sqrt(0.5) * Diagonal(cholesky(C₀).L)), sqrt(0.5) * ones(D), S)
algs[:gpf] = GPF(rand(MvNormal(μ₀, C₀), S), NGmu)
algs[:gf] = GF(copy(μ₀), Matrix(cholesky(C₀).L), S, NGmu)
algs[:iblr] = IBLR(copy(μ₀), inv(C₀), S)

# algs[:spm] = SPM(copy(μ₀), inv(cholesky(C₀).L), S)
# algs[:ngd] = NGD(copy(μ₀), cholesky(C₀).L)


ELBOs = Dict()
times = Dict()
opts = Dict()
for (name, alg) in algs
    ELBOs[name] = zeros(T+1)
    ELBOs[name][1] = ELBO(alg, logπ, nSamples = Stest)
    times[name] = 0
    opts[name] = RMSProp(0.1)
end


opt = Optimiser(ADAM(1.0), IncreasingRate(0.1, 1e-4))
opt = Optimiser(LogLinearIncreasingRate(0.1, 1e-6, 100), ClipNorm(sqrt(D)))
opt = Optimiser( Descent(0.01), ClipNorm(1.0))
opt = Optimiser(InverseDecay(), ClipNorm(1.0))
opt = Optimiser(Descent(0.01), InverseDecay())
opt = Momentum(0.001)
# opt = ScalarADADelta(0.9)
opt = Descent(0.01)
opts[:gpf] = MatADAGrad(0.1)
opts[:gpf] = MatRMSProp(0.1)
opts[:iblr] = Descent(0.1)
# opt = ADAM(0.1)
@showprogress for i in 1:T
    for (name, alg) in algs
        t = @elapsed update!(alg, logπ, opts[name])
        times[name] += t
        ELBOs[name][i+1] = ELBO(alg, logπ, nSamples = Stest)
    end
end
for (name, alg) in algs
    @info "$name :\nELBO = $(ELBO(alg, logπ, nSamples = Stest))\nTime : $(times[name])"
end

## Plotting difference
using Plots
pyplot()
p_L = plot(title = "Free Energy", yaxis=:log)
for (name, alg) in algs
    cut = findfirst(x->x==0, ELBOs[name])
    cut = isnothing(cut) ? T : cut - 1
    plot!(p_L, 1:cut, -ELBOs[name][1:cut], lab = acs[name], color=dcolors[name])
end

p_L |> display
# savefig(plotsdir("Classification - " * @savename(S, NGmu) * ".png"))
## Plot the final status
lim = 20
xrange = range(-lim, lim, length = 300)
yrange = range(-lim, lim, length = 300)
ps = []
logπ([minimum(xrange), minimum(yrange)])
ptruth = contour(xrange, yrange, (w1, w2)->logπ([w1, w2]), title = "truth", colorbar=false)
# ptruth = heatmap(xrange, yrange, (w1, w2)->logπ([w1, w2]), title = "truth", colorbar=false)
# plike = contour!(xrange, yrange, (w1, w2)->logpdf(likelihood(x' * [w1, w2]), y), title = "likelihood", colorbar=false)
# pprior = contour!(xrange, yrange, (w1, w2)->logpdf(prior, [w1, w2]), title = "prior", colorbar=false)
push!(ps, ptruth)
for (name, alg) in algs
    d = MvNormal(alg)
    p_contour = contour(xrange, yrange, (x,y)->logpdf(d, [x, y]), title = acs[name], colorbar=false)
    if alg isa GPF
        scatter!(p_contour, eachrow(alg.X)..., lab="", msw=0.0, alpha = 0.6, color=1)
    end
    push!(ps, p_contour)
end
plot(ps...) |> display



## Show the data and separation
p_data = scatter(eachrow(x)..., group=Int.(y))
for (name, alg) in algs
    m = mean(alg)
    plot!(p_data, x->- x * m[1] / m[2], label=acs[name], color=dcolors[name])
end
p_data |> display
## Showing evolution 

# q = GPF(rand(MvNormal(μ₀, C₀), S), NGmu)
# a = Animation()
# opt = Momentum(0.1)
# @showprogress for i in 1:200
#     p = contour(xrange, yrange, logbanana, title = "i=$i", colorbar=false)
#     contour!(p, xrange, yrange, (x,y)->logpdf(MvNormal(q), [x, y]), colorbar=false)
#     scatter!(p, eachrow(q.X)..., lab="", msw=0.0, alpha = 0.9)
#     frame(a)
#     update!(q, logπ, opt)
# end
# gif(a, plotsdir("Banana - Momentum.gif"), fps = 20)