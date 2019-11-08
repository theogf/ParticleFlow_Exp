using KernelFunctions
using Plots
using LinearAlgebra
using Distributions
using Optim
using StatsFuns: logistic
include("pflowbase.jl")
@info "Starting "
xrange = range(-1.0,1.0,length=200)
X = xrange
p_x = zeros(size(X))
K = kernelmatrix(SqExponentialKernel(10.0),reshape(X,:,1),obsdim=1)+1e-5I
invK = inv(K)
y = 10*rand(MvNormal(K+0.01I))
y = sign.(y)
scatter(X,y)
avoid_inf(x) = max(x,eps(x))
avoid_inf(0.0)
σ = 0.01

phi(x,invK) =  0.5*(dot(x-y,x-y)/σ + dot(x,invK*x))
phi(x,invK) =  0.5*(sum(logpdf.(TDist(3),x-y)) + dot(x,invK*x))
phi(x,invK) =  (sum(-log.(avoid_inf.(logistic.(x.*y)))) + 0.5*dot(x,invK*x))
function free_energy(phi,xs,l::AbstractVector)
    invK = inv(exp(l[1])*kernelmatrix(SqExponentialKernel(exp(l[2])),reshape(X,:,1),obsdim=1)+1e-5I)
    free_energy(phi,xs,invK)
end
function free_energy(phi,xs,invK::AbstractMatrix)
    return -0.5*logdet(cov(xs,dims=2)) + expec(phi,xs,invK)
end
# expec(phi,x_t,invK)
# free_energy(phi,x_t,[1.0,0.1])
#
# grad_hyper_free_energy([1.0,2.0],x_t)
# f(y)
x_0 = optimize(x->phi(x,invK),zero(y),LBFGS())
@info "Optimization done"
plot!(X,x_0.minimizer)
M = 500
# x_init = rand(MvNormal(x_0.minimizer,1.0),M)
x_init = rand(MvNormal(zero(y),0.01),M)
function cb(x_t,invK,t)
    if t%10 == 0
        L = free_energy(phi,x_t,invK)
        p = scatter(X,y,lab="data")
        μ,Σ = mu_and_sig(x_t)
        for k in 1:1:size(x_t,2)
            plot!(X,x_t[:,k],lab="",alpha=0.2)
        end
        plot!(X,μ.+2*sqrt.(diag(Σ)),fillrange=μ.-2*sqrt.(diag(Σ)),alpha=0.1,color=:black,lab="")
        plot!(X,μ,lab="Prediction",lw=3.0,title="t=$t,L=$L")
        p |> display
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
##
x_t = copy(x_init)
X_t = []

θ = [log(10.0),log(10.0)]
η=1.0
opt_x=[VanillaGradDescent(η=η),VanillaGradDescent(η=η)];opt_θ=Adam(α=0.001)
move_particles(x_t,θ,f,100,opt_x,opt_θ,cb=cb2)
# heatmap(K,yflip=true)
x_t
mu_and_sig(x_t)
f_x = mapslices(x->f(x,invK),x_t,dims=1)
m= mean(x_t,dims=2)
sum_var_f(x_t,f_x,m)
add_var_f(x_t,f_x,m)
sum_f(f_x)
x_t.-m
