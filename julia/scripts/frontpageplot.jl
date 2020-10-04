using Makie
using AdvancedVI
using Distributions, LinearAlgebra
using ForwardDiff: gradient
m1 = [0, 0]
m2 = [0.8, 0.8]
d1 = MvNormal(m1, [1 0; 0 1])
d2 = MvNormal(m2, [1 -1.5; -1.5 3])

K = 20
x_init = rand(d1, K)
x_0 = copy(x_init)
xrange = range(-5, 11, length = 500)
function is_std(d, nσ)
    θ = range(0, 2π, length = 100)
    return mean(d) .+ sqrt(nσ) * cholesky(cov(d)).L * [cos.(θ) sin.(θ)]'
end

function is_std(d, nσ, x, y)
    Float64(abs(pdf(d, [x, y]) - pdf(d, mean(d)) * exp(-nσ^2/ 2)) < 0.001)
end
is_std(d1, 1)
## Compute gradients
g = mapslices(x_0; dims=1) do x
    gradient(x) do y
        -logpdf(d2, y)
    end
end

mean_g = -0.1 * vec(mean(g, dims= 2))
xgrads = -0.1 * g*(x_init.-mean(x_init, dims = 2))'/K * (x_init.-mean(x_init,dims=2))
xgrads .+= mean_g
## plotting
scene = Scene()
scatter!(scene, eachrow(x_init)...)
lines!(eachrow(is_std(d1, 1))...)
lines!(eachrow(is_std(d1, 2))...)
lines!(eachrow(is_std(d1, 3))...)
lines!(eachrow(is_std(d2, 1))...)
lines!(eachrow(is_std(d2, 2))...)
lines!(eachrow(is_std(d2, 3))...)
arrows!([first(mean(d1))], [last(mean(d1))],  [first(mean_g)], [last(mean_g)])
arrows!(x_init[1,:], x_init[2,:], xgrads[1,:], xgrads[2,:])
