using Turing
using AdvancedVI; const AVI = AdvancedVI
using KernelFunctions, Distances
using ForwardDiff
using LinearAlgebra
using DistributionsAD, Bijectors
using Flux

x = randn(2000)

@model model(x) = begin
    s ~ InverseGamma(2, 3)
    m ~ Normal(0.0, sqrt(s))
    for i = 1:length(x)
        x[i] ~ Normal(m, sqrt(s))
    end
end
using Makie, Colors

##

max_iter = 1000
m = model(x)
advi = AdvancedVI.ADVI(100,max_iter)
bs = Stacked(inv.(bijector.([InverseGamma(),Normal()])),[1:1,2:2])
# q = AVI.SamplesMvNormal(randn(100,2),[true,false])
# q = AdvancedVI.vi(m, flowvi, 100, optimizer = ADAGrad(0.1))
# @profiler q = AdvancedVI.vi(m, flowvi, 100, optimizer = ADAGrad(0.1))
# global q = AdvancedVI.vi(m, steinvi, q, optimizer = ADAGrad(0.1))
##
logπ = Turing.Variational.make_logjoint(m)
vars = keys(Turing.VarInfo(m).metadata)
q = TuringDiagMvNormal(zeros(2), ones(2))
# q = TuringDenseMvNormal(zeros(2), Diagonal(ones(2)))
q = transformed(q,bs)
q0 = deepcopy(q)
# q = AdvancedVI.vi(logπ, advi, q, optimizer = ADAGrad(0.1))
limits = FRect2D((.5,-.5),(1,1))
t = Node(1)
trajectory = lift(t; init = [Point2f0(q.transform(q0.dist.m))]) do t
        push!(trajectory[], Point2f0(q.transform(q0.dist.m)))
    end
# alpha = lift(t; init = [1.0]) do t
    # push!(alpha[],exp(-t))
# end
sample = lift(t; init = [Point2f0(q.transform(q0.dist.m))]) do t
    sample = [Point2f0(q.transform(q0.dist.m))]
end
scene = Scene(limits=limits)
colors = colormap("Blues",max_iter)
cc = lift(t->colors[1:t],t)
lines!(trajectory,color=cc)
scatter!(sample, color=:blue,markersize=0.02)
xlabel!(scene,"μ_$(vars[1])")
ylabel!(scene,"μ_$(vars[2])")
record(scene,joinpath(@__DIR__,"path_particles_adflow.gif"),framerate=10) do io
    function cb(q,θ,i)
        t[] = i
        global q0 = AVI.update(q,θ)
        recordframe!(io)
    end
    global q = AdvancedVI.vi(logπ, advi, q, optimizer = ADAGrad(0.5), callback = cb)
end
