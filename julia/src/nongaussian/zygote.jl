using DrWatson
@quickactivate

using DataFrames
using BSON
using Flux
using Plots
using Distributions
include(srcdir("utils", "tools.jl"))

setadbackend(Val(:forward_diff))
m0 = [0.1, 0]
m1 = zeros(2)
m2 = zeros(2)

logzygote(x1, x2) = log(zygote(x1, x2))
logzygote(x) = log(zygote(x))
zygote(x1, x2) = zygote([x1, x2])
zygote(x) = 0.5 * pdf(MvNormal(m1, I), x) + 0.5 * pdf(MvNormal(m2, I), x)
function std_line(d, nσ)
    θ = range(0, 2π, length = 100)
    return mean(d) .+ sqrt(nσ) * cholesky(cov(d) .+ 1e-5).L * [cos.(θ) sin.(θ)]'
end
n_particles = [2, 5, 10, 20, 50, 100]
lim = 8
xrange = range(-lim, lim, length = 200)
yrange = range(-lim, lim, length = 200)
totσ = 3
d_init = MvNormal(zeros(2))
ps = []
# for n_p in n_particles

## Running the n_particles
n_p = 20
a = Animation()
η = 0.1
m1 = zeros(2)
m2 = zeros(2)
q = SamplesMvNormal(randn(2, n_p) .+ m0)
@progress for i in 1:40
    q = SamplesMvNormal(randn(2, n_p))
    qvi = AVI.PFlowVI(200, false, false)
    vi(logzygote, qvi, q; optimizer = Descent(0.01))

    ## Plotting
    p = plot(title = "$n_p particles", showaxis = false, legend = false)
    contourf!(p, xrange, yrange, zygote, colorbar = false)
    scatter!(p, eachrow(q.x)...)
    for i in 1:totσ
        plot!(p, eachrow(std_line(q, i))..., color = :white, linewidth = 0.8)
    end
    display(p)
    frame(a)
    global m1 .+= η
    global m2 .-= η
end
gif(a, plotsdir("nongaussian", "zygote.gif"), fps = 10)
## Same thing with VI
m1 = zeros(2)
m2 = zeros(2)
θ = vcat(zeros(2), [1, 0, 1])
q = TuringDenseMvNormal(zeros(2), Diagonal(ones(2)))
a = Animation()
@progress for i in 1:40
    θ = vcat(m0, [1, 0, 1])

    qvi = ADVI(5000, 10)
    vi(logzygote, qvi, q, θ, optimizer = Descent(0.1))
    q = AVI.update(q, θ)
    p = plot(title = "Standard VI", showaxis =false)
    contourf!(p, xrange, yrange, zygote, colorbar = false)
    # scatter!(p, eachrow(q.x)..., label="")
    for j in 1:totσ
        plot!(p, eachrow(std_line(q, j))..., color = :white, label="", linewidth = 0.5)
    end
    frame(a)
    global m1 .+= η
    global m2 .-= η
end
gif(a, plotsdir("nongaussian", "zygote_stdvi.gif"), fps = 10)

# push!(ps, p)
# # end
# plot(ps..., layout = (length(ps), 1), figsize = (800, 2))
# savefig(plotsdir("nongaussian", "banana.png"))
