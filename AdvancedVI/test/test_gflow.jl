using AdvancedVI
const AVI = AdvancedVI
using Distributions
using Flux
using LinearAlgebra
using ProgressMeter

n_dim = 2
d_target = MvNormal(0.5 * ones(n_dim), randn(n_dim, n_dim) |> x->x * x')
mutable struct RobbinsMonro
    κ::Float64
    τ::Float64
    state::IdDict
end
  
function RobbinsMonro(κ::Real = 0.51, τ::Real = 1)
    @assert 0.5 < κ <= 1 "κ should be in the interval (0.5,1]"
    @assert τ > 0 "τ should be positive"
    RobbinsMonro(κ, τ, IdDict())
end

function Flux.Optimise.apply!(o::RobbinsMonro, x, Δ)
    κ = o.κ
    τ = o.τ
    n = get!(o.state, x, 1)
    Δ .*= 1 / (τ + n)^κ
    o.state[x] = n + 1
    return Δ
end
logπ(x) = logpdf(d_target, x)
## Running algorithm
AVI.setadbackend(:forward_diff)
# μ = Vector(mean(d_target))
μ = rand(n_dim)
μ = -2 * ones(n_dim)
Γ = Matrix(1.0 * I(n_dim))
# Γ = reshape([1.0, 0.5], n_dim, :)
#rand(n_dim, n_dim)
S = 3
algs = Dict(
    # :gflow => AVI.GaussFlow(1, S, false, false),
    # :gpflow => AVI.GaussPFlow(1, true, false),
    # :dsvi => AVI.DSVI(1, S),
    :iblr => AVI.IBLR(1, S, :rep),
    # :fcs => AVI.FCS(1, S),
)

function MvNormal(q::AVI.AbstractPosteriorMvNormal)
    MvNormal(mean(q), cov(q)+ 1e-5I)
end

## Testing full rank
fullqs = Dict(
    :gflow => AVI.LowRankMvNormal(copy(μ), copy(Γ)),
    :gpflow => AVI.SamplesMvNormal(rand(MvNormal(μ, Γ * Γ'), S)),
    :dsvi => AVI.CholMvNormal(copy(μ), cholesky(Γ).L),
    :iblr => AVI.PrecisionMvNormal(copy(μ), inv(Γ)),
    :fcs => AVI.FCSMvNormal(copy(μ), Matrix(cholesky(Γ).L) / sqrt(2), ones(length(μ)) / sqrt(2))
)
opt = Descent(0.1)
# opt = [Descent(0.1), RobbinsMonro(0.99, 50)]
# opt = ADAGrad(0.1)
using Plots
xlin = range(-10, 10, length = 100)
ylin = range(-10, 10, length = 100)
a = Animation()
@showprogress for i in 1:1000
    Plots.contour(xlin, ylin, (x,y)->pdf(d_target, [x,y]), clims = (0, 0.2), color = :red, colorbar = false, title = "i = $i")
    for (name, alg) in algs
        q = fullqs[name]
        alg isa AVI.GaussPFlow && scatter!(eachrow(q.x)..., label="")
        Plots.contour!(xlin, ylin, (x,y)->pdf(MvNormal(q), [x,y]))
        AVI.vi(logπ, alg, q, optimizer = opt)
    end
    frame(a)
end
gif(a)
## Testing sampling
ps = Dict()
for (name, alg) in algs
    q = mfqs[name]
    p = contour(xlin, ylin, (x,y)->pdf(MvNormal(q), [x,y]), color = :blue, colorbar=false, title=name)
    scatter!(eachrow(rand(q, 100))..., label="")
    ps[name] = p
end
plot(values(ps)...)

## Testing mean-field
mfqs = Dict(
    :gflow => AVI.MFMvNormal(copy(μ), diag(Γ)),
    :gpflow => AVI.MFSamplesMvNormal(rand(MvNormal(μ, Γ * Γ'), S)),
    :dsvi => AVI.MFMvNormal(copy(μ), diag(Γ)),
    :iblr => AVI.DiagPrecisionMvNormal(copy(μ), inv.(diag(Γ)))
)
opt = Descent(0.1)
# opt = [Descent(0.1), RobbinsMonro(0.99, 50)]
# opt = ADAGrad(0.1)

using Plots
xlin = range(-10, 10, length = 100)
ylin = range(-10, 10, length = 100)
a = Animation()
@showprogress for i in 1:100 
    contour(xlin, ylin, (x,y)->pdf(d_target, [x,y]), clims = (0, 0.2), color = :red, colorbar = false, title = "i = $i")
    for (name, alg) in algs
        q = mfqs[name]
        alg isa AVI.GaussPFlow && scatter!(eachrow(q.x)...)
        contour!(xlin, ylin, (x,y)->pdf(MvNormal(q), [x,y]))
        AVI.vi(logπ, alg, q, optimizer = opt)
    end
    frame(a)
end
gif(a)

## Testing sampling
ps = Dict()
for (name, alg) in algs
    q = mfqs[name]
    p = contour(xlin, ylin, (x,y)->pdf(MvNormal(q), [x,y]), color = :blue, colorbar=false, title=name)
    scatter!(eachrow(rand(q, 100))..., label="")
    ps[name] = p
end
plot(values(ps)...)