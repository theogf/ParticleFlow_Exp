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
