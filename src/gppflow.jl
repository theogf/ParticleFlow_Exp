using DrWatson
quickactivate(joinpath(@__DIR__,".."))
include("pflowbase.jl")
using Plots
using Optim
using AugmentedGaussianProcesses
using StatsFuns: logistic
@info "Starting "
xrange = range(-1.0,1.0,length=200)
xrange = rand(Uniform(-1.0,1.0),100)
xsort= sortperm(xrange)
X = reshape(xrange,:,1)
p_x = zeros(size(X))
kernel = SqExponentialKernel(10.0)
K = kernelmatrix(kernel,X,obsdim=1)+1e-5I
invK = inv(K)
variance = 10.0
σ = 0.1
y_reg = variance*rand(MvNormal(K+σ*I))
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
p_reg = GPModel((x,y)->sum(logpdf.(Normal.(y,sqrt(σ)),x)),kernel,variance,X,y_reg,gradll=(x,p)->gradlogpdf.(Normal.(p.y,sqrt(σ)),x))

xmap = optimize(p_reg,x->∇phi(x,p_reg),randn(length(y_reg)),LBFGS(),Optim.Options(iterations=100,f_tol=1e-8),inplace=false)
@info "Optimization done"
m = GP(collect(xrange)[xsort],y_reg[xsort],kernel,noise=σ,opt_noise=false,variance=variance)
train!(m,100)
μgp, siggp = predict_f(m,vec(X)[xsort],covf=true)
scatter(sort(xrange),y_reg[xsort],lab="data")
plot!(sort(xrange),Optim.minimizer(xmap)[xsort],lab="MAP",lw=3.0)
plot!(sort(xrange),μgp,ribbon=2*sqrt.(siggp),lab="GP",lw=3.0)
## Training with Gaussian Noise
M = 300
x_init = rand(MvNormal(xmap.minimizer,0.01),M)
# x_init = rand(MvNormal(zero(y_reg),0.01),M)
x_t = copy(x_init)
X_t = []
η=0.01
opt_x=[VanillaGradDescent(η=η),VanillaGradDescent(η=η)]
move_particles(x_t,p_reg,100,opt_x)
μp,sigp = mu_and_sig(x_t)
sigp = diag(sigp)
plot!(xrange[xsort],μp[xsort],ribbon=2*sqrt.(sigp),lw=3.0,lab="Particles")

################################################
## Optimizing for Student-T Noise
ν= 3
p_stu = GPModel((x,y)->sum(logpdf.(TDist(ν),x-y)),kernel,variance,X,y_reg,gradll=(x,p)->gradlogpdf.(TDist(ν),x-p.y))

m = VGP(collect(xrange),y_reg,kernel,StudentTLikelihood(Float64(ν)),AnalyticVI())
train!(m,10)
μgp,siggp = predict_f(m,collect(xrange)[xsort],covf=true)
#MAP
x0 = randn(length(y_reg))
xmap = optimize(p_stu,x0,LBFGS(),Optim.Options(iterations=1000,x_tol=1e-5,f_tol=1e-8),inplace=false)
@info "Optimization done"
heatmap(inv(p_stu.invK.mat)[xsort,xsort])
scatter(xrange[xsort],y_reg[xsort],lab="Data")
plot!(xrange[xsort],Optim.minimizer(xmap)[xsort],lab="MAP",lw=3.0)
plot!(xrange[xsort],μgp,ribbon=2*sqrt.(max.(siggp,0.0)),lab="GP",lw=3.0)
## Training with Student-T Noise
M = 200

x_init = rand(MvNormal(xmap.minimizer,0.01),M)
# x_init = rand(MvNormal(zero(y_reg),0.01),M)
x_t = copy(x_init)
X_t = []
η=0.01
opt_x=[VanillaGradDescent(η=η),VanillaGradDescent(η=η)]
move_particles(x_t,p_stu,10,opt_x)
μp,sigp = mu_and_sig(x_t)
sigp = diag(sigp)
plot!(xrange[xsort],μp[xsort],ribbon=2*sqrt.(sigp),lw=3.0,lab="Particles")
################################################
## Optimizing for Classification
p_log = GPModel((x,y)->sum(log.(avoid_inf.(logistic.(x.*y)))),kernel,variance,X,y_log,gradll=(x,p)->p.y.*logistic.(-p.y.*x))
xmap = optimize(p_log,x->∇phi(x,p_log),randn(length(y_log)),LBFGS(),Optim.Options(iterations=100,x_tol=1e-5,f_tol=1e-8),inplace=false)
@info "Optimization done"
m = VGP(collect(xrange),y_log,kernel,LogisticLikelihood(),AnalyticVI())
train!(m,10)
μgp,siggp = predict_f(m,collect(xrange)[xsort],covf=true)

scatter(X,y_log,lab="data")
plot!(xrange[xsort],Optim.minimizer(xmap)[xsort],lab="MAP",lw=3.0)
plot!(xrange[xsort],μgp,ribbon=2*sqrt.(siggp),lab="MAP",lw=3.0,fillalpha=0.2)
## Training with Student-T Noise
M = 200

# x_init = rand(MvNormal(x_0.minimizer,0.01),M)
x_init = rand(MvNormal(zero(y_reg),0.01),M)
x_t = copy(x_init)
X_t = []
η=0.5
opt_x=[VanillaGradDescent(η=η),VanillaGradDescent(η=η)]
cb(x_t,p_log,0)
move_particles(x_t,p_log,10,opt_x)
