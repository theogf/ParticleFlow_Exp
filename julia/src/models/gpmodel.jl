mutable struct GPModel <: AbstractModel
    log_likelihood::Function
    grad_log_likelihood::Function
    kernel::Kernel
    σ::Vector{Float64}
    params
    X::AbstractArray
    y::Vector
    m::Vector
    C::Matrix
    invK
    opt
end
jitter = 1e-3

function GPModel(log_likelihood,kernel,variance,X,y;gradll=nothing,opt=Flux.ADAM(0.001),params...)
    if isnothing(gradll)
        gradll = (x,p)->ForwardDiff.gradient(X->p.log_likelihood(X,p.y),x)
    end
    K = kernelmatrix(kernel,X,obsdim=1)
    invK = inv(variance*(K+jitter*median(diag(K))*I))
    GPModel(log_likelihood,gradll,kernel,[variance],collect(params),X,y,similar(y),Matrix(undef,length(y),length(y)),invK,opt)
end

log_gp_prior(x,invK) = -0.5*(dot(x,invK*x)+length(x)*log(2π)-logdet(invK))
(p::GPModel)(x) = phi(x,p,p.params,p.invK)
grad_log_gp_prior(x,invK) = -(invK*x)
phi(x,p::GPModel,params,invK) = -p.log_likelihood(x,p.y) - log_gp_prior(x,invK)
function free_energy(x,p::GPModel)
    C = cov(x,dims=2)
    -0.5*logdet(C+jitter*tr(C)/size(C,1)*I) + expec(x,p,p.params,p.invK)
end
expec(x,p::GPModel,params,invK) = sum(phi(x[:,i],p,params,invK) for i in 1:size(x,2))/size(x,2)
_f(x,p::GPModel) = -0.5*∇phi(x,p)
∇phi(x,p::GPModel) = -(p.grad_log_likelihood(x,p)+grad_log_gp_prior(x,p.invK))
function predic_f(p,x::Matrix,x_test)
    p.m, p.C = m_and_C(x)
    k_star = first(p.σ)*kernelmatrix(p.kernel,x_test,p.X,obsdim=1)
    μf = k_star*p.invK*p.m
    A = p.invK*(I-p.C*p.invK)
    k_starstar = first(p.σ)*(kerneldiagmatrix(p.kernel,x_test,obsdim=1).+jitter)
    Σf = k_starstar - AugmentedGaussianProcesses.opt_diag(k_star*A,k_star)
    return μf,Σf
end

function predic_f(p,x::Vector,x_test)
    k_star = first(p.σ)*kernelmatrix(p.kernel,x_test,p.X,obsdim=1)
    return k_star*p.invK*x
end


function hyper_grad(x,p::GPModel)
    ForwardDiff.gradient(θ->expec(x,p,θ,inv(θ[1]*(kernelmatrix(base_kernel(p.kernel)(θ[2:end]),p.X,obsdim=1)+jitter*I))),p.params)
end
# function update_params!(p::GPModel,x,opt)
#     ps = Flux.params(p.kernel)
#     push!(ps,p.σ)
#     return ∇ = Zygote.gradient(()->expec(x,p,p.params,inv(first(p.σ)*(kernelmatrix(p.kernel,p.X,obsdim=1)+1e-5I))),ps)
#     for p in ps
#         p .= exp.(log.(p).+Flux.Optimise.apply!(opt,p,p.*∇[p]))
#     end
#     p.invK = inv(first(p.σ)*(kernelmatrix(p.kernel,p.X,obsdim=1)+1e-5I))
# end

function update_params!(p::GPModel,x,opt)
    ps = Flux.params(p.kernel)
    push!(ps,p.σ)
    return ∇ = Zygote.gradient(()->inv(first(p.σ)*(kernelmatrix(p.kernel,p.X,obsdim=1)+jitter*I))|>K->0.5*(dot(x,K*x)/size(x,2)+logdet(K)),ps)
    for p in ps
        p .= exp.(log.(p).+Flux.Optimise.apply!(opt,p,p.*∇[p]))
    end
    p.invK = inv(first(p.σ)*(kernelmatrix(p.kernel,p.X,obsdim=1)+jitter*I))
end
