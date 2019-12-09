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
