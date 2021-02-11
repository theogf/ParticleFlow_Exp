using AdvancedVI
using Distributions
using Plots
using ProgressMeter

target = MvNormal(zeros(2), [1 0.5; 0.5 2])
logπ(x) = logpdf(target, x)

max_iter = 100
n_particles = 20
k = transform(SqExponentialKernel(),1.0)
alg = AdvancedVI.SVGD(1, k)
q = AVI.EmpiricalDistribution(randn(2, n_particles) * 2 .+ 1)
opt = Descent(1.0)

a = Animation()
xlin = range(-5, 5, length = 100)
ylin = range(-5, 5, length = 100)
fr_to_save = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
j=0
@showprogress for i in 1:max_iter
    if i ∈ fr_to_save
        Plots.contour(xlin, ylin, (x,y)->pdf(target, [x,y]), clims = (0, 0.2), color = :red, colorbar = false, title = "i = $i")
        Plots.scatter!(eachrow(q.x)..., label="")
        # Plots.contour!(xlin, ylin, (x,y)->pdf(MvNormal(q), [x,y]))
        AVI.vi(logπ, alg, q, optimizer = opt)
        frame(a)
        savefig("/tmp/svgd_viz$j.png")
        j += 1
    end
end
gif(a, fps=2)