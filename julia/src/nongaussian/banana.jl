using DataFrames
using BSON
using Flux
using Plots
include(srcdir("train_model.jl"))
include(srcdir("utils", "tools.jl"))

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
ps = []
n_p = 100
q = SamplesMvNormal(randn(2, n_p))
qvi = AVI.PFlowVI(2000, false, false)

q_final = vi(x->-logbanana(x), qvi, q; optimizer = ADAGrad(0.1))
p = plot()
contourf!(p, xrange, yrange, banana)
scatter!(p, eachrow(q.x)...)
for i in 1:totσ
    l = std_line(q, i)
    plot!(p, eachrow(l)..., color = :white, label="", linewidth = 0.3)
end
display(p)
# for n_p in n_particles
