using Turing
using AdvancedVI; const AVI = AdvancedVI
using Distances
using ForwardDiff
using LinearAlgebra
using DistributionsAD, Bijectors
using Flux
using Makie, Colors

x = randn(2000)

@model model(x) = begin
    s ~ InverseGamma(2, 3)
    m ~ Normal(0.0, sqrt(s))
    for i = 1:length(x)
        x[i] ~ Normal(m, sqrt(s))
    end
end

##

max_iter = 400
m = model(x)
flowvi = AdvancedVI.PFlowVI(max_iter, true, true)
advi = AdvancedVI.ADVI(10,max_iter)
# q = AVI.SamplesMvNormal(randn(100,2),[true,false])
# q = AdvancedVI.vi(m, flowvi, 100, optimizer = ADAGrad(0.1))
# @profiler q = AdvancedVI.vi(m, flowvi, 100, optimizer = ADAGrad(0.1))
# global q = AdvancedVI.vi(m, steinvi, q, optimizer = ADAGrad(0.1))
##
nParticles = 1000
b = bijector(m)
q = transformed(SamplesMvNormal(randn(2,nParticles)),b)
q.dist.μ
logπ = Turing.Variational.make_logjoint(m)
# q = AdvancedVI.vi(logπ, flowvi, q, optimizer = ADAGrad(0.1))
AVI.update_q!(q.dist)
limits = FRect2D((.5,-.5),(1,1))
t = Node(1)
trajectories = [lift(t; init = [Point2f0(q.transform(q.dist.x[:,i]))]) do t
        push!(trajectories[i][], Point2f0(q.transform(q.dist.x[:,i])))
end  for i in 1:q.dist.n_particles ]
trajectorymean = lift(t; init = [Point2f0(q.transform(q.dist.μ))]) do t
        push!(trajectorymean[], Point2f0(q.transform(q.dist.μ)))
    end
# alpha = lift(t; init = [1.0]) do t
    # push!(alpha[],exp(-t))
# end
samples = lift(t; init = Point2f0.(q.transform.(eachcol(q.dist.x)))) do t
    samples =Point2f0.(q.transform.(eachcol(q.dist.x)))
end
meansample = lift(t; init = [Point2f0(q.transform(q.dist.μ))]) do t
    meansample[] = [Point2f0(q.transform(q.dist.μ))]
end

scene = Scene(limits=limits)

colors = colormap("Reds",max_iter)
colormean = colormap("Blues",max_iter)
cc = lift(t->colors[1:t],t)
ccmean = lift(t->colormean[1:t],t)
lines!.(trajectories,color=cc)
scatter!(samples, color=:red, markersize=0.01)
lines!(trajectorymean,color=ccmean)
scatter!(meansample, color=:blue, markersize=0.02)

record(scene,joinpath(@__DIR__,"path_particles_pflow.gif"),framerate=10) do io
    function cb(q,i)
        t[] = i
        recordframe!(io)
    end
    global q = AdvancedVI.vi(logπ, flowvi, q, optimizer = ADAGrad(1.0), callback = cb)
end
