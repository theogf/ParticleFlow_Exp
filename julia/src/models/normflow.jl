using Bijectors
using Flux

mutable struct NormFlowModel <: AbstractModel
    logprior::Function
    loglikelihood::Function
    bijector::Bijector
    params::Vector
    m::Vector
    C::Matrix
    opt
end

function NormFlowModel(logprior,loglikelihood,bijector,params,opt=Flux.ADAM())
    NormFlowModel(logprior,loglikelihood,bijector,params,Vector{Float64}(undef,0),Matrix{Float64}(undef,0,0),opt)
end

(p::NormFlowModel)(x) = phi(collect(x),p,p.params,p.bijector)
phi(x,p::NormFlowModel,params,bij) =  - _loglike(p,params,forward(bij,x)...) - p.logprior(x,params)
_loglike(p::NormFlowModel,params,z_k,detJ) = p.loglikelihood(z_k,params) + detJ
function free_energy(x,p::NormFlowModel)
    C = cov(x,dims=2)
    -0.5*logdet(C+1e-5I*tr(C)/size(C,1)) + expec(x,p,p.params,p.bijector)
end
# expec(x,p::NormFlowModel,params,bij) = reduce(+,phi.(eachcol(x),[p],[params],[bij]))/size(x,2)
expec(x,p::NormFlowModel,params,bij) = sum(phi(x[:,i],p,params,bij) for i in 1:size(x,2))/size(x,2)
# expec(x,p::NormFlowModel,params,bij) = reduce(+,p.loglikelihood(bij(x),params))/size(x,2)
# expec(x,p::NormFlowModel,params,bij) = reduce(+,logabsdetjac.(Ref(bij),eachcol(x)))/size(x,2)
# expec(x,p::NormFlowModel,params,bij) = mean(phi.(eachcol(x),[p],[params],[bij]))
z_k(x,p::NormFlowModel) = p.bijector(x)
q_0(x,p::NormFlowModel) = MvNormal(m_and_C(x)...)
q_k(x,p::NormFlowModel) = transformed(q_0(x,p),p.bijector)
_f(x,p::NormFlowModel) = -0.5*∇phi(x,p)
∇phi(x,p::NormFlowModel) = ForwardDiff.gradient(p,x)
hyper_grad(x,p::NormFlowModel,ps) = Zygote.gradient(()->expec(x,p,p.params,p.bijector),ps)
Flux.@functor PlanarLayer
Flux.@functor RadialLayer
Flux.@functor Composed
function update_params!(p::NormFlowModel,x,opt)
    ps = Flux.params(p.bijector)
    ∇ = hyper_grad(x,p,ps)
    Flux.Optimise.update!(opt,ps,∇)
    # for par in ps
        # Flux.Optimise.update!(opt,par,∇[par])
    # end
end
