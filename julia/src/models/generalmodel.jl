mutable struct GeneralModel <: AbstractModel
    logprior::Function
    loglikelihood::Function
    params::Vector
    m::Vector
    C::Matrix
    opt
end

function GeneralModel(logprior,loglikelihood,params,opt=Flux.ADAM(0.001))
    GeneralModel(logprior,loglikelihood,params,[],Matrix{Float64}(undef,0,0),opt)
end

(p::GeneralModel)(x) = phi(x,p,p.params)
phi(x,p::GeneralModel,params) = -p.loglikelihood(x,params) - p.logprior(x,params)
function free_energy(x,p::GeneralModel)
    C = cov(x,dims=2)
    -0.5*logdet(C+1e-5I*tr(C)/size(C,1)) + expec(x,p,p.params)
end
expec(x,p::GeneralModel,params) = mean(phi.(eachcol(x),[p],[params]))
g(x,p::GeneralModel) = ∇phi(x,p)
∇phi(x,p::GeneralModel) = ForwardDiff.gradient(x->p(x),x)
hyper_grad(x,p::GeneralModel) = ForwardDiff.gradient(θ->mean(expec.(x,p,θ)),p.params)
function update_params!(m::GeneralModel,x,opt)
    if length(m.params) > 0
        ps = params(m.params)
        ∇ = gradient(()->m(x),ps)
        for p in ps
            p .-= apply!(p.opt,p,∇)
        end
    end
end
