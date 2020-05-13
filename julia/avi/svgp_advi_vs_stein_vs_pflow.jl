using DrWatson

using AdvancedVI; const AVI = AdvancedVI
using Turing
using LinearAlgebra
using Distributions, DistributionsAD
using Makie, StatsMakie, Colors, MakieLayout, CairoMakie
CairoMakie.activate!()
using KernelFunctions, Flux, KernelDensity
@info "Packages Loaded!"

N = 1000
M = 25
x = range(0, 1, length = N)
Z = range(0,1, length = M)
θ = log.([1.0, 10.0, 1e-3])
k = exp(θ[1]) * transform(SEKernel(), exp(θ[2]))
K = kernelmatrix(k, x) + 1e-5I
f = rand(MvNormal(K))
y = f + randn(N) * exp(θ[3])
plot(x, y)
##
# xrange = range(min(mu_init,mu1)-3.0,max(mu_init,mu2)+3.0,length=300)

# Makie.plot!(ax, x, f)
# Makie.scatter!(ax, x, y)

nParticles = 60
max_iters = 2
n_sig = 2

function proj(μ, Σ, x, Z, θ)
    k = exp(θ[1]) * transform(SEKernel(), exp(θ[2]))
    Kmm = kernelmatrix(k, Z) + 1e-5I
    invKmm = inv(Kmm)
    Kxm = kernelmatrix(k, x, Z)
    Kxx = kerneldiagmatrix(k, x) .+ 1e-5
    κ = Kxm * invKmm
    Σf = Kxx + diag(κ *(Σ * invKmm - I) * Kxm')
    return κ * μ, Σf
end

setadbackend(:reverse_diff)
function meta_logπ(θ)
    k = exp(θ[1]) * transform(SEKernel(), exp(θ[2]))
    Kmm = kernelmatrix(k, Z) + 1e-5I
    invKmm = inv(Kmm)
    Kxm = kernelmatrix(k, x, Z)
    Kxx = kerneldiagmatrix(k, x) .+ 1e-5
    κ = Kxm * invKmm
    K̃ = Kxx - diag(κ * Kxm')
    d = TuringDenseMvNormal(zeros(size(Z, 1)), Kmm)
    return z -> sum(logpdf.(Normal.(y, exp(θ[3]) .+ K̃), κ * z)) + logpdf(d, z)
end
logπ_reduce = meta_logπ(θ)
logπ_reduce(rand(M))

mu_init = randn(M)
sig_init = exp.(randn(M))
D = Matrix(Diagonal(sig_init))
L_init = Matrix(cholesky(D).L)
##
# advi = Turing.Variational.ADVI(nParticles, max_iters)
advi = AVI.ADVI(nParticles, max_iters)
θ_init = vcat(mu_init, L_init[:])
adq = AVI.transformed(TuringDenseMvNormal(mu_init, L_init*L_init'), AVI.Bijectors.Identity{1}())
θ_ad = θ .- 1

quadvi = AVI.ADQuadVI(nParticles, max_iters)
# θ_init = vcat(mu_init, L_init[:])
quadq = AVI.transformed(TuringDenseMvNormal(mu_init, L_init*L_init'), AVI.Bijectors.Identity{1}())

steinvi = AVI.SteinVI(max_iters, transform(SqExponentialKernel(), 1.0))
steinq =
    AVI.SteinDistribution(rand(MvNormal(mu_init, sig_init), nParticles))
θ_stein = θ .- 1

gaussvi = AVI.PFlowVI(max_iters, false, true)
gaussq = SamplesMvNormal(rand(MvNormal(mu_init, sig_init), nParticles))
θ_gauss = θ .- 1

# gaussq = AVI.transformed(SamplesMvNormal(rand(Normal(mu_init, sqrt(sig_init)),1,nParticles)),AVI.Bijectors.Identity{1}())

α =  1.0
optad = ADAGrad(α)
optquad = ADAGrad(α)
optstein = ADAGrad(α)
optgauss = ADAGrad(α)
hp_opt = ADAM(0.01)

t = Node(0)
m_C_ad = lift(t) do t
    μ = adq.dist.m
    Σ = Matrix(adq.dist.C)
    m, C = proj(μ, Σ, x, Z, θ_stein)
end
meanad = lift(m_C_ad) do x
    x[1]
end
stdadplus = lift(m_C_ad) do x
    x[1] + n_sig *sqrt.(x[2])
end
stdadminus = lift(m_C_ad) do x
    x[1] - n_sig * sqrt.(x[2])
end

meanquad = lift(t) do t
    quadq.dist.m
end
m_C_stein = lift(t) do t
    μ = mean(steinq)
    Σ = cov(steinq)
    m, C = proj(μ, Σ, x, Z, θ_stein)
end

meanstein = lift(m_C_stein) do m_C
    m_C[1]
end

stdsteinplus = lift(m_C_stein) do m_C
    m_C[1] + n_sig * sqrt.(m_C[2])
end

stdsteinminus = lift(m_C_stein) do m_C
    m_C[1] - n_sig * sqrt.(m_C[2])
end

m_C_gauss = lift(t) do t
    μ = mean(gaussq)
    Σ = cov(gaussq)
    m, C = proj(μ, Σ, x, Z, θ_gauss)
end

meangauss = lift(m_C_gauss) do m_C
    m_C[1]
end

stdgaussplus = lift(m_C_gauss) do m_C
    m_C[1] + n_sig * sqrt.(m_C[2])
end

stdgaussminus = lift(m_C_gauss) do m_C
    m_C[1] - n_sig * sqrt.(m_C[2])
end
##
scene, layout = layoutscene(1, resolution = (800, 400))
ax = layout[1, 1] = LAxis(scene)
# ax.tellheight = true
ls = [Makie.plot!(ax, x, f, color=:red, linewidth=7.0),
    Makie.plot!(ax, x, meanad, linewidth = 3.0, color = :green),
    # Makie.plot!(ax, x, meanquad, linewidth = 3.0, color = :orange),
    Makie.plot!(ax, x, meanstein, linewidth = 3.0, color = :blue),
    Makie.plot!(ax, x, meangauss, linewidth = 3.0, color = :purple),
    ]

band!(ax, x, stdadplus, stdadminus, color = RGBA(colorant"green", 0.3), transparency =true)
band!(ax, x, stdgaussplus, stdgaussminus, color = RGBA(colorant"purple", 0.3), transparency =true)
band!(ax, x, stdsteinplus, stdsteinminus, color = RGBA(colorant"blue", 0.3), transparency =true)
# Makie.scatter!(ax, gaussxs, zeros(gaussq.n_particles), msize = 1.0)
leg = layout[1,1] = LLegend(scene,ls,
                    ["p(x)",
                    "ADVI",
                    # "Quad VI",
                    "Stein VI",
                    "Gaussian Particles",
                    ],
                    tellheight = false,
                    tellwidth=  false,
                    width=Auto(false),height=Auto(false),
                    halign = :left, valign = :top,
                    margin = (10,10,10,10)
                )
scene


##
record(scene,joinpath(@__DIR__,"svgp_ad_vs_steinflow.gif"),framerate=25) do io
    for i in 1:100
        # global adq = AVI.vi(logπ_reduce, advi, adq, θ_init, optimizer = optad)
        # global quadq = AVI.vi(logπ_base, quadvi, quadq, θ_init, optimizer = optquad)
        # AVI.vi(logπ_reduce, steinvi, steinq, optimizer = optstein)
        # AVI.vi(logπ_reduce, gaussvi, gaussq, optimizer = optgauss)
        global adq = AVI.vi(
            meta_logπ,
            advi,
            adq,
            θ_init,
            optimizer = optad,
            hyperparams = θ_ad,
            hp_optimizer = hp_opt,
        )
        AVI.vi(
            meta_logπ,
            steinvi,
            steinq,
            optimizer = optstein,
            hyperparams = θ_stein,
            hp_optimizer = hp_opt,
        )
        AVI.vi(
            meta_logπ,
            gaussvi,
            gaussq,
            optimizer = optgauss,
            hyperparams = θ_gauss,
            hp_optimizer = hp_opt,
        )
        @show exp.(θ_gauss)
        @show exp.(θ_ad)
        t[] = i
        recordframe!(io)
    end
end
##

##
# AVI.vi(
#     meta_logπ,
#     gaussvi,
#     gaussq,
#     optimizer = optgauss,
#     hyperparams = θ_init,
#     hp_optimizer = hp_opt,
# )
#
# @profiler AVI.vi(
#         meta_logπ,
#         gaussvi,
#         gaussq,
#         optimizer = optgauss,
#         hyperparams = θ_init,
#         hp_optimizer = hp_opt,
#     )
