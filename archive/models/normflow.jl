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

(m::NormFlowModel)(x) = phi(collect(x),m,m.params,m.bijector)
phi(x,m::NormFlowModel,params,bij) =  - _loglike(m,params,forward(bij,x)...) - m.logprior(m,params)
_loglike(m::NormFlowModel,params,z_k,detJ) = m.loglikelihood(z_k,params) + detJ
function free_energy(x,m::NormFlowModel)
    C = cov(x, dims=2)
    - 0.5 * logdet(C + 1e-5I * tr(C) / size(C, 1)) + expec(x, m, m.params, m.bijector)
end
# expec(x,p::NormFlowModel,params,bij) = reduce(+,phi.(eachcol(x),[p],[params],[bij]))/size(x,2)
expec(x, m::NormFlowModel, params, bij) = sum(phi(x[:,i], m, params, bij) for i in 1:size(x,2)) / size(x,2)
# expec(x,p::NormFlowModel,params,bij) = reduce(+,p.loglikelihood(bij(x),params))/size(x,2)
# expec(x,p::NormFlowModel,params,bij) = reduce(+,logabsdetjac.(Ref(bij),eachcol(x)))/size(x,2)
# expec(x,p::NormFlowModel,params,bij) = mean(phi.(eachcol(x),[p],[params],[bij]))
z_k(x,m::NormFlowModel) = m.bijector(x)
q_0(x,m::NormFlowModel) = MvNormal(m_and_C(x)...)
q_k(x,m::NormFlowModel) = transformed(q_0(x,m),m.bijector)
g(x,m::NormFlowModel) = ∇phi(x,m)
∇phi(x, m::NormFlowModel) = ForwardDiff.gradient(m,x)
hyper_grad(x, m::NormFlowModel,ps) = Zygote.gradient(()->expec(x, m , m.params, m.bijector),ps)
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
