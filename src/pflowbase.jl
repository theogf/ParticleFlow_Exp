using ForwardDiff, LinearAlgebra
using GradDescent, Zygote
using KernelFunctions
using Distributions
using PDMats

abstract type AbstractModel end

struct GeneralModel <: AbstractModel
    logprior::Function
    loglikelihood::Function
    params::Vector
    opt::Optimizer
end

function GeneralModel(logprior,loglikelihood,params,opt=Adam(α=0.001))
    GeneralModel(logprior,loglikelihood,params,opt)
end

(p::GeneralModel)(x) = phi(x,p,p.params)
phi(x,p::GeneralModel,params) = -p.loglikelihood(x,params) - p.logprior(x,params)
function free_energy(x,p::GeneralModel)
    C = cov(x,dims=2)
    -0.5*logdet(C+1e-5I*tr(C)/size(C,1)) + expec(x,p,p.params)
end
expec(x,p::GeneralModel,params) = mean(phi.(eachcol(x),[p],[params]))
_f(x,p::GeneralModel) = -0.5*∇phi(x,p)
∇phi(x,p::GeneralModel) = ForwardDiff.gradient(x->p(x),x)
hyper_grad(x,p::GeneralModel) = ForwardDiff.gradient(θ->mean(expec.(x,p,θ)),p.params)
function update_params!(p::GeneralModel,x)
    if length(p.params) > 0
        ∇ = hyper_grad(x,p)
        p.params .+= update(p.opt,∇)
    end
end

mutable struct GPModel <: AbstractModel
    log_likelihood::Function
    grad_log_likelihood::Function
    kernel::Kernel
    params::Vector{Float64}
    X::AbstractArray
    y::Vector
    invK
    opt::Optimizer
end

function GPModel(log_likelihood,kernel,variance,X,y;gradll=nothing,opt=Adam(α=0.001))
    if isnothing(gradll)
        gradll = (x,p)->ForwardDiff.gradient(X->p.log_likelihood(X,p.y),x)
    end
    K = kernelpdmat(kernel,X,obsdim=1)
    invK = inv(variance*(K+1e-5*mean(diag(K))*I))
    GPModel(log_likelihood,gradll,kernel,vcat(variance,first(KernelFunctions.params(kernel))),X,y,invK,opt)
end

log_gp_prior(x,invK) = -0.5*(dot(x,invK*x)+length(x)*log(2π)-logdet(invK))
(p::GPModel)(x) = phi(x,p,p.params,p.invK)
grad_log_gp_prior(x,invK) = -(invK*x)
phi(x,p::GPModel,params,invK) = -p.log_likelihood(x,p.y) - log_gp_prior(x,invK)
function free_energy(x,p::GPModel)
    C = cov(x,dims=2)
    -0.5*logdet(C+1e-5*tr(C)/size(C,1)*I) + expec(x,p,p.params,p.invK)
end
expec(x,p::GPModel,params,invK) = mean(phi.(eachcol(x),[p],[params],[invK]))
_f(x,p::GPModel) = -0.5*∇phi(x,p)
∇phi(x,p::GPModel) = -(p.grad_log_likelihood(x,p)+grad_log_gp_prior(x,p.invK))
function hyper_grad(x,p::GPModel)
    ForwardDiff.gradient(θ->        expec(x,p,θ,inv(θ[1]*(kernelmatrix(base_kernel(p.kernel)(θ[2:end]),p.X,obsdim=1)+1e-5I))),p.params)
end
function update_params!(p::GPModel,x)
    ∇ = hyper_grad(x,p)
    p.params .= exp.(log.(p.params).+update(p.opt,p.params.*∇))
    KernelFunctions.set_params!(p.kernel,p.params[2:end])
    p.invK = inv(p.params[1]*(kernelmatrix(p.kernel,p.X,obsdim=1)+1e-5I))
end

## General Helper  ###
@inline sum_f(f) = mean(f,dims=2)
@inline add_var_f(x,f,m) = (0.5I + sum_var_f(x,f,m))*(x.-m)
@inline sum_var_f(x,f,m) = mean(eachcol(f).*transpose.(eachcol(x.-m)))
@inline mu_and_sig(x) = vec(mean(x,dims=2)),cov(x,dims=2)+1e-5I
no_prior(x,p) = zero(eltype(x))
base_kernel(k::Kernel) = eval(nameof(typeof(k)))

function move_particles(x,p,T=10,opt_x=[Momentum(η=0.001),Momentum(η=0.001)];cb=nothing,Xt=nothing)
    L = free_energy(x,p)
    L_new = Inf
    x_new = copy(x)
    @progress for iter in 1:T
        m = mean(x,dims=2)
        f_x = mapslices(y->_f(y,p),x,dims=1)
        Δx = update(opt_x[1],sum_f(f_x))
        var_x = update(opt_x[2],add_var_f(x,f_x,m))
        x_new .= x .+ Δx .+ var_x
        L_new = free_energy(x_new,p)
        α = 0.5
        while L_new > L
            x_new .= x .+ α*Δx .+ α*var_x
            L_new = free_energy(x_new,p)
            α *= 0.1
            if α < 1e-10
                @error "α too small, skipping step!"
                α = 0.0
                x_new .= x .+ α*Δx .+ α*var_x
                break
            end
        end
        x .= x_new
        update_params!(p,x)
        L = L_new
        if !isnothing(Xt)
            push!(X_t,copy(x_t))
        end
        if !isnothing(cb)
            cb(x,p,iter)
        end
    end
end

function iseverylog10(i)
    i%(10^(floor(Int64,log10(i))))==0
end
