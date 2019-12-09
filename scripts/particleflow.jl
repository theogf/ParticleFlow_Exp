using DrWatson
quickactivate(joinpath(@__DIR__,".."))
include(srcdir("pflowbase.jl"))
using Makie


# Two modes
p1 = GeneralModel(no_prior,(x,p)->-0.5*(x[1]^2*x[2]^2+x[1]^2+x[2]^2-8x[1]-8x[2]),[])

# Normal Gaussian
μ = zeros(2); Σ = rand(2,2)|> x->x*x'#diagm(ones(2))
d_base = MvNormal(μ,Σ)
p2 = GeneralModel(no_prior,(x,p)->logpdf(d_base,collect(x)),[])

##
mu0 = [0.0,0.0]#zeros(2)
C = diagm(ones(2))
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
## Objects for animations
titlestring = Node("t=0")

## Training two modes
M = 3
x_init = rand(MvNormal(mu0,C),M)
x_t1 = copy(x_init)
X_t = []
xrange = range(-2.5,7.5,length=100)
X = Iterators.product(xrange,xrange)
p_x_target = zeros(size(X))
for (i,x) in enumerate(X)
    p_x_target[i] = exp(-p1(x))
end
##
x_p = Node(x_t1)
m,C = m_and_C(x_t1)
dist_x = lift(x->MvNormal(m_and_C(x)...),x_p)
p_X = lift(x->pdf.(Ref(x),collect.(X)),dist_x)
x_p1  = lift(x->x[1,:],x_p)
x_p2  = lift(x->x[2,:],x_p)
normalizer(dist_x.val)
levels = lift(x->Float32.(normalizer(x)*exp.(-0.5(5:-1:1))),dist_x)
## MWE
using Makie, Distributions
xrange = range(-2.5,7.5,length=100)
X = Iterators.product(xrange,xrange)
d1 = MvNormal(zeros(2),I)
p_x = pdf.(Ref(d1),collect.(X))
scene = contour(xrange,xrange,p_x,levels=10,fillrange=true,linewidth=0.0)
d2 = MvNormal([-5,5],I)
p_x2 = pdf.(Ref(d2),collect.(X))
levels = inv(det(cov(d2))*2π)*exp.(-0.5(5:-1:1)) # Return levels at 1:5 sigmas
scene = contour!(scene,xrange,xrange,p_x2,color=:white,levels=levels)
# AbstractPlotting.inline!(true)
##
scene = contour(xrange,xrange,p_x_target,levels=100,fillrange=true,linewidth=0.0)
scene = contour!(xrange,xrange,p_X,color=:white,levels=levels)
scene = scatter!(x_p1,x_p2,color=:red,markersize=0.3)
scene = title(scene,titlestring)


T = 100; fps = 10
opt_x1 =[Momentum(η=0.1),Momentum(η=0.1)]

record(scene,joinpath(plotsdir(),"gifs","2modes_$(M)_particles.gif"),1:T,framerate=fps) do i
    push!(titlestring,"t=$i")
    move_particles(x_t1,p1,opt_x1)
    push!(x_p,x_t1)
end

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
