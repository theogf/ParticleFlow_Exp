using DrWatson
@quickactivate


using Turing
using Distributions, DistributionsAD
using AdvancedVI; const AVI = AdvancedVI
using Makie, StatsMakie, Colors, MakieLayout
using KernelFunctions, Flux, KernelDensity

mu1 = -2.0;
mu2 = 3.0
mu_init = -10.0
sig_init = 1.0
d1 = Normal(mu1)
d2 = Normal(mu2)
d = MixtureModel([d1, d2],[1/3, 2/3])

x = rand(d,1000)
xrange = range(min(mu_init,mu1)-3.0,max(mu_init,mu2)+3.0,length=300)
scene, layout = layoutscene()
ax = layout[1, 1] = LAxis(scene)
Makie.plot!(ax,histogram(nbins=100,normalize=true), x, color=RGBA(0.0,0.0,0.0,0.5))
p = Makie.plot!(ax,xrange,pdf.(Ref(d),xrange),color=:red, linewidth=7.0)

nParticles = 100
max_iters = 2


##
# advi = Turing.Variational.ADVI(nParticles, max_iters)
advi = AVI.ADVI(nParticles, max_iters)
θ_init = [mu_init,sig_init]
adq = AVI.transformed(TuringDiagMvNormal([mu_init],[sig_init]),AVI.Bijectors.Identity{1}())

quadvi = AVI.ADQuadVI(nParticles, max_iters)
θ_init = [mu_init,sig_init]
quadq = AVI.transformed(TuringDiagMvNormal([mu_init],[sig_init]),AVI.Bijectors.Identity{1}())

steinvi = AVI.SteinVI(max_iters, transform(SqExponentialKernel(), 1.0))
steinq =
    AVI.SteinDistribution(rand(Normal(mu_init, sqrt(sig_init)), 1, nParticles))

gaussvi = AVI.PFlowVI(max_iters, false, true)
gaussq = SamplesMvNormal(rand(Normal(mu_init, sqrt(sig_init)),1,nParticles))
# gaussq = AVI.transformed(SamplesMvNormal(rand(Normal(mu_init, sqrt(sig_init)),1,nParticles)),AVI.Bijectors.Identity{1}())

using ReverseDiff
setadbackend(:reversediff)
logπ_base(x) = log(1/3*pdf(d1,first(x)) + 2/3*pdf(d2,first(x)))

α =  1.0
optad = ADAGrad(α)
optquad = ADAGrad(α)
optstein = ADAGrad(α)
optgauss = ADAGrad(α)

t = Node(0)
pdfad = lift(t) do _
    adqn = Normal(adq.dist.m[1], sqrt(adq.dist.σ[1]))
    pdfad = pdf.(Ref(adqn),xrange)
end
pdfquad = lift(t) do _
    quadqn = Normal(quadq.dist.m[1], sqrt(quadq.dist.σ[1]))
    pdfquad = pdf.(Ref(quadqn),xrange)
end
pdfstein = lift(t) do _
    steinqn = kde(steinq.x[:])
    pdfstein = pdf.(Ref(steinqn),xrange)
end
pdfgauss = lift(t) do _
    if gaussq.Σ[1,1] > 0
        gaussqn = Normal(gaussq.μ[1],sqrt(gaussq.Σ[1,1]))
        pdfgauss = pdf.(Ref(gaussqn),xrange)
    end
end
gaussxs = lift(t) do _
    gaussq.x[:]
end

scene, layout = layoutscene()
ax = layout[1, 1] = LAxis(scene))
Makie.plot!(ax,histogram(nbins=100,normalize=true), x, color=RGBA(0.0,0.0,0.0,0.5))

plots_with_labels = [
    ("p(x)", Makie.plot!(ax,xrange,pdf.(Ref(d),xrange),color=:red, linewidth=7.0)),
    ("ADVI", Makie.plot!(ax,xrange, pdfad, linewidth = 3.0, color = :green)),
    #("Quad VI", Makie.plot!(ax,xrange, pdfquad, linewidth = 3.0, color = :orange)),
    ("Stein VI", Makie.plot!(ax,xrange, pdfstein, linewidth = 3.0, color = :blue)),
    ("Gaussian Particles", Makie.plot!(ax,xrange, pdfgauss, linewidth = 3.0, color = :purple)),
]

labels, line_plots = zip(plots_with_labels...)
Makie.scatter!(ax, gaussxs, zeros(gaussq.n_particles), msize = 1.0)
leg = LLegend(
    scene, collect(line_plots), collect(labels),
    width=Auto(false), height=Auto(false),
    halign = :left, valign = :top, margin = (10,10,10,10)
)
scene

##
record(scene,joinpath(@__DIR__,"path_particles_ad_vs_steinflow.gif"),framerate=25) do io
    for i in 1:500
        global adq = AVI.vi(logπ_base, advi, adq, θ_init, optimizer = optad)
        # the following doesn't seem to compile
        #global quadq = AVI.vi(logπ_base, quadvi, quadq, θ_init, optimizer = optquad)
        AVI.vi(logπ_base, steinvi, steinq, optimizer = optstein)
        AVI.vi(logπ_base, gaussvi, gaussq, optimizer = optgauss)
        t[] = i
        recordframe!(io)
    end
end
