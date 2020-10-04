using DrWatson
@quickactivate

using DataFrames
using BSON
using Flux
using Plots
pyplot()
using AdvancedVI, Distributions
include(srcdir("utils", "tools.jl"))
Distributions.mean(d::TuringDenseMvNormal) = d.m
Distributions.cov(d::TuringDenseMvNormal) = d.C.L * d.C.U
logbanana(x1, x2) = -0.5 * (0.01 * x1^2 + (x2 + 0.1x1^2 - 10)^2)
logbanana(x) = logbanana(first(x), last(x))
banana(x1, x2) = exp(logbanana(x1, x2))
banana(x) = exp(logbanana(x))
function std_line(d, nσ)
    θ = range(0, 2π, length = 100)
    return mean(d) .+ sqrt(nσ) * cholesky(cov(d) .+ 1e-5).L * [cos.(θ) sin.(θ)]'
end
n_particles = [2, 5, 10, 20, 50, 100]
xrange = range(-30, 30, length = 200)
yrange = range(-60, 20, length = 200)
totσ = 3
d_init = MvNormal(zeros(2))
ps = Vector{Any}(undef, length(n_particles)+1)
n_p = 1000
## Training the particles
for (i, n_p) in enumerate(n_particles)
    q = SamplesMvNormal(randn(2, n_p))
    qvi = AVI.PFlowVI(2000, false, false)


    vi(logbanana, qvi, q; optimizer = ADAGrad(0.1))

    ## Plotting
    p = plot(title = "$n_p particles", showaxis =false)
    contourf!(p, xrange, yrange, banana, colorbar = false)
    scatter!(p, eachrow(q.x)..., label="")
    for i in 1:totσ
        l = std_line(q, i)
        plot!(p, eachrow(l)..., color = :white, label="", linewidth = 0.3)
    end
    display(p)
    ps[i] = p
end
## Training classic VI
θ = vcat(zeros(2), [1, 0, 1])
q = TuringDenseMvNormal(zeros(2), Diagonal(ones(2)))
qvi = ADVI(2000, 100)
vi(logbanana, qvi, q, θ)
q = AVI.update(q, θ)
p = plot(title = "Standard VI", showaxis =false)
contourf!(p, xrange, yrange, banana, colorbar = false)
# scatter!(p, eachrow(q.x)..., label="")
for i in 1:totσ
    l = std_line(q, i)
    plot!(p, eachrow(l)..., color = :white, label="", linewidth = 0.5)
end
display(p)
ps[end] = p
## Plot all results
p = plot(ps..., layout = (1, length(ps)), size = (1000, 200), dpi = 300)
display(p)
ispath(plotsdir("nongaussian")) ? nothing : mkpath(plotsdir("nongaussian"))
savefig(plotsdir("nongaussian", "banana.png"))
