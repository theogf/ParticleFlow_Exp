using DrWatson
quickactivate(joinpath(@__DIR__,".."))
include("pflowbase.jl")
using Plots; pyplot()
using StatsPlots


# Two modes
p1 = GeneralModel(no_prior,(x,p)->-0.5*(x[1]^2*x[2]^2+x[1]^2+x[2]^2-8x[1]-8x[2]),[])

# Normal Gaussian
μ = zeros(2); Σ = rand(2,2)|> x->x*x'#diagm(ones(2))
d_base = MvNormal(μ,Σ)
p2 = GeneralModel(no_prior,(x,p)->logpdf(d_base,collect(x)),[])

##
mu0 = [0.0,0.0]#zeros(2)
C = diagm(ones(2))
# m = mean(x_t,dims=2)
# f_t = mapslices(f,x_t,dims=1)
# sum_f(f_t)
function cb(x,p::GeneralModel,t)
    if iseverylog10(t)
        L = free_energy(x,p)
        m,S = mu_and_sig(x)
        σs = 4:-1:1
        pl = contourf!(xrange,xrange,p_x',levels=100,lw=0.0,title="t=$t, L=$L",cbar=false)
        levels = inv((2π)*det(S))*exp.(-0.5*σs)
        d = MvNormal(m,S)
        p_q = zeros(size(X))
        for (i,x) in enumerate(X)
            p_q[i] = pdf(d,collect(x))
        end
        scatter!(pl,eachrow(x)...,lab="")
        contour!(pl,xrange,xrange,p_q',levels=levels,clabels=["$(i)σ" for i in σs],
        lab="",color=:white)
        frame(anim)
    end
end
## Training two modes
M = 3
x_init = rand(MvNormal(mu0,0.01*C),M)
x_t1 = copy(x_init)
X_t = []
xrange = range(-2.5,7.5,length=100)
X = Iterators.product(xrange,xrange)
p_x = zeros(size(X))
for (i,x) in enumerate(X)
    p_x[i] = exp(-p1(x))
end
anim = Animation()
opt_x1 =[Momentum(η=0.01),Momentum(η=0.01)]
move_particles(x_t1,p1,1000,opt_x1,cb=cb)
gif(anim,plotsdir("gifs","2modes_M=$M.gif"),fps=8) |> display

## Training a Gaussian
M = 3
x_init = rand(MvNormal(mu0,0.01*C),M)
x_t2 = copy(x_init)
X_t = []
xrange = range(-4.0,4.0,length=100)
X = Iterators.product(xrange,xrange)
p_x = zeros(size(X))
for (i,x) in enumerate(X)
    p_x[i] = exp(-p2(x))
end
p_base = contourf(xrange,xrange,p_x',levels=100,lw=0.0,title="",cbar=false,grid=:none,axis=:none)
anim = Animation()
opt_x2 =[Momentum(η=0.01),Momentum(η=0.01)]
p_x
move_particles(x_t2,p2,1000,opt_x2,cb=cb)
gif(anim,plotsdir("gifs","gaussian_M=$M.gif"),fps=8) |> display


# function multi_contour()
xrange = range(-4.0,4.0,length=100)
X1 = xrange*xrange'
X2 = 1e10*sin.(-xrange*xrange')
contourf(xrange,xrange,X1',levels=100,colorbar=false)
twinx()
contour!(xrange,xrange,X2',color=:white,levels=[-1e10,0,1e-10],colorbar=false)
# end
multi_contour()
