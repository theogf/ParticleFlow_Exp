using DrWatson
quickactivate(joinpath(@__DIR__,".."))
include(srcdir("pflowbase.jl"))
include(srcdir("makie_plotting.jl"))

# Two modes
p1 = GeneralModel(no_prior,(x,p)->-0.5*(x[1]^2*x[2]^2+x[1]^2+x[2]^2-8x[1]-8x[2]),[])

# Normal Gaussian
μ = zeros(2); Σ = rand(2,2)|> x->x*x'#diagm(ones(2))
d_base = MvNormal(μ,Σ)
p2 = GeneralModel(no_prior,(x,p)->logpdf(d_base,collect(x)),[])

# Neal's Funnel

dz = Normal(0,3)
dx(z) = Normal(0,exp(z/2))
grad_log_normal_z(x,z) = -0.5(z+x^2*exp(-z))
p3 = GeneralModel(no_prior,(x,p)->logpdf(dz,x[2])+grad_log_normal_z(x[1],x[2]),[])


##
mu0 = [0.0,0.0]#zeros(2)
C0 = diagm(ones(2))

## Training two modes
M = 100
x_init = rand(MvNormal(mu0.+[2.0,0.0],C0),M)
x_t1 = copy(x_init)
X_t = []
xrange = range(-2.5,7.5,length=100)
X = Iterators.product(xrange,xrange)
p_x_target = zeros(size(X))
for (i,x) in enumerate(X)
    p_x_target[i] = exp(-p1(x))
end
xrange = range(-2.5,7.5,length=100)
scene,x_p,tstring = set_plotting_scene_2D(x_t1,xrange,xrange,p_x_target)
T = 100; fps = 10
opt_x1 =[Descent(0.1),Descent(0.1)]

record(scene,joinpath(plotsdir(),"gifs","2modes_$(M)_particles.gif"),1:T,framerate=fps) do i
    push!(tstring,"t=$i")
    move_particles(x_t1,p1,opt_x1,precond_b=true,precond_A=true)
    push!(x_p,x_t1)
end

## Training a Gaussian
M = 3
x_init = rand(MvNormal(mu0,0.01*C0),M)
x_t2 = copy(x_init)
X_t = []
xrange = range(-4.0,4.0,length=100)
X = Iterators.product(xrange,xrange)
p_x_target = zeros(size(X))
for (i,x) in enumerate(X)
    p_x_target[i] = exp(-p2(x))
end
opt_x2 =[Descent(0.0),Descent(1.0)]
scene,x_p,tstring = set_plotting_scene_2D(x_t2,xrange,xrange,p_x_target)
T = 100; fps = 10

record(scene,joinpath(plotsdir(),"gifs","2modes_$(M)_particles.gif"),1:T,framerate=fps) do i
    push!(tstring,"t=$i")
    move_particles(x_t2,p2,opt_x2,precond_A=true)
    push!(x_p,x_t2)
end

## Training on funnel

M = 3
x_init = rand(MvNormal(zeros(2),I),M)
x_t3 = copy(x_init)
X_t = []
xrange = range(-6.0,4.0,length=100)
X = Iterators.product(xrange,xrange)
p_x_target = zeros(size(X))
for (i,x) in enumerate(X)
    p_x_target[i] = exp(-p3(x))
end
opt_x3 =[ADAM(0.1),ADAM(0.1)]
scene,x_p,tstring = set_plotting_scene_2D(x_t3,xrange,xrange,p_x_target)
T = 100; fps = 10
record(scene,joinpath(plotsdir(),"gifs","2modes_$(M)_particles.gif"),1:T,framerate=fps) do i
# for i in 1:T
    push!(tstring,"t=$i")
    move_particles(x_t3,p3,opt_x3,precond_A=!false,precond_b=!false)
    push!(x_p,x_t3)
end
