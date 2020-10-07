using DrWatson
@quickactivate

using DataFrames
using BSON
using Flux, LinearAlgebra
using Plots, ColorSchemes
pyplot()
using AdvancedVI, Distributions, DistributionsAD
const AVI = AdvancedVI
include(srcdir("utils", "tools.jl"))
Distributions.mean(d::TuringDenseMvNormal) = d.m
Distributions.cov(d::TuringDenseMvNormal) = d.C.L * d.C.U
logbanana(x1, x2) = -0.5(0.01 * x1^2 + 0.1(x2 + 0.1x1^2 - 10)^2)
logbanana(x) = logbanana(first(x), last(x))
banana(x1, x2) = exp(logbanana(x1, x2))
banana(x) = exp(logbanana(x))
# contourf(xrange, yrange, banana, colorbar = false)
opt = ADAGrad(0.1)
opt = Flux.Optimise.Optimiser(ClipNorm(10.0), Descent(1.0))
opt = Flux.Optimise.Optimiser(ClipNorm(10.0), Momentum(0.1))
# opt = Momentum(0.01)
function std_line(d, nσ)
    θ = range(0, 2π, length = 100)
    return mean(d) .+ sqrt(nσ) * cholesky(cov(d) .+ 1e-5).L * [cos.(θ) sin.(θ)]'
end
n_particles = [3, 5, 10, 20, 50]
xrange = range(-30, 30, length = 200)
yrange = range(-40, 20, length = 200)
totσ = 3
d_init = MvNormal(zeros(2))
ps = Vector{Any}(undef, length(n_particles) + 2)
n_p = 1000
## Training the particles
for (i, n_p) in enumerate(n_particles)
    global q = SamplesMvNormal(randn(2, n_p))
    qvi = AVI.PFlowVI(2000, false, false)

    vi(logbanana, qvi, q; optimizer = deepcopy(opt))

    ## Plotting
    p = plot(title = "$n_p particles", showaxis =false, xlims = extrema(xrange), ylims= extrema(yrange))
    contourf!(p, xrange, yrange, banana, colorbar = false)
    scatter!(p, eachrow(q.x)..., label="")
    for i in 1:totσ
        plot!(p, eachrow(std_line(q, i))..., color = :white, label="", linewidth = 0.8)
    end
    display(p)
    ps[i] = p
end
## Training classic VI
θ = vcat(zeros(2), [1, 0, 1])
q = TuringDenseMvNormal(zeros(2), Diagonal(ones(2)))
qvi = ADVI(2000, 100)
vi(logbanana, qvi, q, θ, optimizer=deepcopy(opt))
q = AVI.update(q, θ)
p = plot(title = "Standard VI", showaxis =false, xlims = extrema(xrange), ylims = extrema(yrange))
contourf!(p, xrange, yrange, banana, colorbar = false)
# scatter!(p, eachrow(q.x)..., label="")
for i in 1:totσ
    plot!(p, eachrow(std_line(q, i))..., color = :white, label="", linewidth = 0.5)
end
display(p)
ps[end] = p

## Training particle VI with different opt


## Training with Stein
using KernelFunctions
# q = SteinDistribution(randn(2, 50))
# qvi = AVI.SteinVI(2000, KernelFunctions.transform(SqExponentialKernel(), 0.1))

# vi(logbanana, qvi, q; optimizer = deepcopy(opt))
global q = SamplesMvNormal(randn(2, 50))
qvi = AVI.PFlowVI(2000, false, false)

vi(logbanana, qvi, q; optimizer = ADAM(0.1))

##
p = plot(title = "ADAM", showaxis =false, xlims = extrema(xrange), ylims = extrema(yrange))
contourf!(p, xrange, yrange, banana, colorbar = false, c=:thermal)
scatter!(p, eachrow(q.x)..., label="", color = :green, msw = 1.0, ms = 7.0)
for i in 1:totσ
    l = std_line(q, i)
    plot!(p, eachrow(l)..., color = :white, label="", linewidth = 1.0)
end
display(p)
ps[end-1] = p

## Plot all results
p = plot(ps..., layout = (1, length(ps)), size = (1000, 200), dpi = 300)
display(p)
ispath(plotsdir("nongaussian")) ? nothing : mkpath(plotsdir("nongaussian"))
savefig(plotsdir("nongaussian", "banana.png"))
