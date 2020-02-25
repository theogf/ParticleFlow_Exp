using DrWatson
quickactivate(joinpath(@__DIR__,".."))
include(srcdir("pflowbase.jl"))
include(srcdir("makie_plotting.jl"))
using Makie, Colors
using Optim
using AugmentedGaussianProcesses
using StatsFuns: logistic
@info "Starting "
xrange = range(-1.0,1.0,length=200)
xrange = sort(rand(Uniform(-1.0,1.0),100))
xpred = range(-1.0,1.0,length=500)
xsort= sortperm(xrange)
X = reshape(xrange,:,1)
p_x = zeros(size(X))
kernel = SqExponentialKernel(10.0)
K = kernelmatrix(kernel,X,obsdim=1)+1e-5I
invK = inv(K)
variance = 1.0
_σ = 0.1
y_reg = variance*rand(MvNormal(K+_σ*I))
y_log = sign.(y_reg)
avoid_inf(x) = max(x,eps(x))

##
function cb(x_t,p,t)
    if t%1 == 0
        L = free_energy(x_t,p)
        fig = scatter(p.X,p.y,lab="data")
        μ,Σ = mu_and_sig(x_t)
        # for k in 1:1:size(x_t,2)
        #     plot!(fig,X,x_t[:,k],lab="",alpha=0.2)
        # end
         plot!(fig,p.X[xsort],μ[xsort],lab="Prediction",ribbon=2*sqrt.(diag(Σ)[xsort]),fillalpha=0.4,lw=3.0,title="t=$t,F=$L")
        fig |> display
    end
end

function cb2(x_t,invK,t)
    m = mean(x_t,dims=2)
    f_x = mapslices(x->f(x,invK),x_t,dims=1)
    Δx = sum_f(f_x)
    var_x = add_var_f(x_t,f_x,m)
    αs = 10.0.^(-5:0.1:0)
    Lold =free_energy(phi,x_t,invK)
    Ls = zero(αs)
    for (i,α) in enumerate(αs)
        x_new = x_t .+ α*Δx .+ α*var_x
        Ls[i] = free_energy(phi,x_new,invK)
    end
    plot(αs,fill(Lold,length(αs)),color=:red)
    plot!(αs,Ls,xaxis=:log,yaxis=:log) |> display
end

## Optimizing for Gaussian Noise
p_reg = GPModel((x,y)->sum(logpdf.(Normal.(y,sqrt(_σ)),x)),deepcopy(kernel),variance,X[xsort,:],y_reg[xsort],gradll=(x,p)->gradlogpdf.(Normal.(p.y,sqrt(_σ)),x),opt=ADAM(0.00))

xmap = optimize(p_reg,x->∇phi(x,p_reg),randn(length(y_reg)),LBFGS(),Optim.Options(iterations=100,f_tol=1e-8),inplace=false)
@info "Optimization done"
m = GP(collect(xrange)[xsort],y_reg[xsort],deepcopy(kernel),noise=_σ,opt_noise=false,variance=variance,optimiser=ADAM(0.001))
train!(m,1)
# μgp, siggp = predict_f(m,vec(X)[xsort],covf=true)
μgp, siggp = predict_f(m,xpred,covf=true)

scatter(sort(xrange),y_reg[xsort],markersize=0.1)
plot!(sort(xrange),Optim.minimizer(xmap)[xsort],linewidth=3.0,color=:blue)
fill_between!(xpred,μgp .- 2*sqrt.(siggp),μgp .+ 2*sqrt.(siggp),where=trues(length(xpred)),color=RGBA(colorant"red",0.3))
plot!(xpred,μgp,linewidth=3.0,color=:black)
# Training with Gaussian Noise
M = 50
# x_init = rand(MvNormal(xmap.minimizer,0.01),M)
x_init = rand(MvNormal(zero(y_reg),0.01),M)
x_t = copy(x_init)
X_t = []
∇f1 = zero(y_reg)
scene, x_p, ∇f_p=  set_plotting_scene_GP(x_t,p_reg,xrange,y_reg,xpred,μgp, siggp,∇f1)

η=0.1
opt_0 = [Flux.Descent(η),Flux.Descent(η)]
opt_x=[Flux.Descent(η),Flux.Descent(η)]
glob_m = mean(x_t,dims=2)[:]
# @progress for i in 1:100
record(scene, plotsdir("gifs","gaussiangp.gif"), 1:91 ; framerate = 10) do i
# for i in 1:92
    @info i
    if i < 50
        move_particles(x_t,p_reg,opt_0,precond_b=true,precond_A=false)
    else
        move_particles(x_t,p_reg,opt_x,precond_b=true,precond_A=false)
    end
    norm(∇f1) |> display
    push!(x_p,x_t)
    push!(∇f_p,∇f1)
    # scene |> display
end

# move_particles(x_t,p_reg,opt_x)
##
μp,sigp = m_and_diagC(x_t)
μf, σf = predic_f(p_reg,x_t,reshape(xpred,:,1))
plot!(xpred,μf,color=:green,linewidth=2.0)
fill_between!(xpred,μf.-sqrt.(σf),μf.+sqrt.(σf),color=:green,where=trues(length(xpred)))

################################################
## Optimizing for Student-T Noise
ν= 3
p_stu = GPModel((x,y)->sum(logpdf.(TDist(ν),x-y)),kernel,variance,X,y_reg,gradll=(x,p)->gradlogpdf.(TDist(ν),x-p.y))

m = VGP(collect(xrange),y_reg,kernel,StudentTLikelihood(Float64(ν)),AnalyticVI())
train!(m,10)
μgp,siggp = predict_f(m,collect(xrange)[xsort],covf=true)
#MAP
x0 = randn(length(y_reg))
xmap = optimize(p_stu,x0,LBFGS(),Optim.Options(iterations=100,x_tol=1e-5,f_tol=1e-8),inplace=false)
@info "Optimization done"
μgp, siggp = predict_f(m,xpred,covf=true)

scatter(xrange[xsort],y_reg[xsort],markersize=0.1)
plot!(sort(xrange),Optim.minimizer(xmap)[xsort],linewidth=3.0,color=:blue)
fill_between!(xpred,μgp .- 2*sqrt.(max.(0.0,siggp)),μgp .+ 2*sqrt.(max.(0.0,siggp)),where=nothing,color=RGBA(colorant"red",0.3))
plot!(xpred,μgp,linewidth=3.0,color=:red)
## Training with Student-T Noise
M = 300
# x_init = rand(MvNormal(xmap.minimizer,1.0),M)
x_init = rand(MvNormal(zero(y_reg),1.0),M)
x_t = copy(x_init)
X_t = []
∇f1 = zero(y_reg)
scene, x_p, ∇f_p=  set_plotting_scene_GP(x_t,xrange,y_reg,xpred,μgp, siggp,∇f1)

η=0.001
opt_0 = [Flux.ADAM(0.1),Flux.ADAM(0.1)]
opt_x= opt_0#[Flux.Descent(η),Flux.Descent(η)]
# @progress for i in 1:100
record(scene, plotsdir("gifs","gaussiangp_studentt.gif"), 1:100; framerate = 10) do i
    @info i
    if i < 50
        move_particles(x_t,p_stu,opt_0,precond_b=!true,precond_A=!false)
    else
        move_particles(x_t,p_stu,opt_x,precond_b=!true,precond_A=!false)
    end
    push!(x_p,x_t)
    push!(∇f_p,∇f1)
end

################################################
## Optimizing for Classification
p_log = GPModel((x,y)->sum(log.(avoid_inf.(logistic.(x.*y)))),kernel,variance,X,y_log,gradll=(x,p)->p.y.*logistic.(-p.y.*x))
xmap = optimize(p_log,x->∇phi(x,p_log),randn(length(y_log)),LBFGS(),Optim.Options(iterations=100,x_tol=1e-5,f_tol=1e-8),inplace=false)
@info "Optimization done"
m = VGP(collect(xrange),y_log,kernel,LogisticLikelihood(),AnalyticVI())
train!(m,10)
μgp,siggp = predict_f(m,xpred,covf=true)

scatter(vec(X),y_log,markersize=0.1)
plot!(xrange[xsort],Optim.minimizer(xmap)[xsort],color=:blue)
fill_between!(xpred,μgp .- 2*sqrt.(max.(0.0,siggp)),μgp .+ 2*sqrt.(max.(0.0,siggp)),where=nothing,color=RGBA(colorant"red",0.3))
plot!(xpred,μgp,lw=3.0,color=:red)
## Training with Classification
M = 300
x_init = rand(MvNormal(xmap.minimizer,1.0),M)
# x_init = rand(MvNormal(zero(y_reg),1.0),M)
x_t = copy(x_init)
X_t = []
∇f1 = zero(y_reg)
scene, x_p, ∇f_p =  set_plotting_scene_GP(x_t,xrange,y_log,xpred,μgp, siggp,∇f1)

η=0.001
opt_0 = [Flux.ADAM(0.1),Flux.ADAM(0.1)]
opt_x= opt_0#[Flux.Descent(η),Flux.Descent(η)]
# @progress for i in 1:100
record(scene, plotsdir("gifs","gaussiangp_classification.gif"), 1:400; framerate = 10) do i
    @info i
    if i < 50
        move_particles(x_t,p_log,opt_0,precond_b=false,precond_A=false)
    else
        move_particles(x_t,p_log,opt_x,precond_b=false,precond_A=false)
    end
    push!(x_p,x_t)
    push!(∇f_p,∇f1)
end
