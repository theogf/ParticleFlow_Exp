using DrWatson
quickactivate(joinpath(@__DIR__,".."))
include(srcdir("pflowbase.jl"))
using Makie


# Two modes
ll = (x,p)->-0.5*(x[1]^2*x[2]^2+x[1]^2+x[2]^2-8x[1]-8x[2])
p1 = NormFlowModel(no_prior,ll,PlanarLayer(2)∘RadialLayer(2),[])
# p1 = NormFlowModel(no_prior,ll,PlanarLayer(2)∘PlanarLayer(2)∘PlanarLayer(2),[])

##
mu0 = [0.0,0.0]#zeros(2)
C = diagm(ones(2))
## Objects for animations
titlestring = Node("t=0, F=Inf")

## Training two modes
M = 200
x_init = rand(MvNormal(mu0,C),M)
x_t1 = copy(x_init)
X_t = []
xrange = range(-2.5,7.5,length=100)
X = Iterators.product(xrange,xrange)
p_x_target = zeros(size(X))
for (i,x) in enumerate(X)
    p_x_target[i] = exp(ll(x,[]))
end
##
x_p = Node(x_t1)
m,C = m_and_C(x_t1)
dist_x0 = lift(x->MvNormal(m_and_C(x)...),x_p)
dist_xk = lift(x->transformed(x,p1.bijector),dist_x0)
p_X0 = lift(x->pdf.(Ref(x),collect.(X)),dist_x0)
# p_Xk = lift(x->exp.(logpdf_with_trans.(Ref(x),collect.(X),true)),dist_xk)
p_Xk = lift(x->exp.(logpdf_forward.(Ref(x),inv(p1.bijector).(collect.(X)))),dist_xk)
# contour(xrange,xrange,logpdf_forward.([dist_xk[]],collect.(X)),fillrange=true)
x_p1  = lift(x->x[1,:],x_p)
x_p2  = lift(x->x[2,:],x_p)
x_pk = lift(x->hcat(p1.bijector.(eachcol(x))...),x_p)
x_p1k  = lift(x->x[1,:],x_pk)
x_p2k  = lift(x->x[2,:],x_pk)
levels = lift(x->Float32.(normalizer(x)*exp.(-0.5(5:-1:1))),dist_x0)
scene = contour(xrange,xrange,p_x_target,levels=100,fillrange=true,linewidth=0.0,colormap=:inferno,padding=(0.0,0.0))
scene = scatter!(x_p1,x_p2,color=:white,markersize=0.3)
scene = scatter!(x_p1k,x_p2k,color=:red,markersize=0.3)
scene = contour!(xrange,xrange,p_X0,color=:white,levels=5)
scene = contour!(xrange,xrange,p_Xk,color=:red,levels=5,linewidth=2.0,padding=(0.0,0.0))
scene = title(scene,titlestring)
T = 100; fps = 10
opt_x1 =[GradDescent.Momentum(η=0.1),GradDescent.Momentum(η=0.1)]

record(scene,joinpath(plotsdir(),"gifs","2modes_$(M)_normflowparticles.gif"),1:T,framerate=fps) do i
    push!(titlestring,"t=$i, F=$(free_energy(x_t1,p1))")
    move_particles(x_t1,p1,opt_x1)
    push!(x_p,x_t1)
end
