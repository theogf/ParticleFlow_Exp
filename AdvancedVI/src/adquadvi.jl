"""
$(TYPEDEF)

Automatic Differentiation Variational Inference (ADVI) with automatic differentiation
backend `AD`.

# Fields

$(TYPEDFIELDS)
"""
struct ADQuadVI{AD, T} <: VariationalInference{AD}
    "Number of points used to estimate the ELBO expectation in each optimization step."
    nPoints::Int
    nodes::Vector{T}
    weights::Vector{T}
    "Maximum number of gradient steps."
    max_iters::Int
end

function ADQuadVI(nPoints::Int=100, max_iters::Int=1000)
    return ADQuadVI{ADBackend(), Float64}(nPoints, gausshermite(nPoints)..., max_iters)
end

alg_str(::ADQuadVI) = "ADQuadVI"

nSamples(alg::ADQuadVI) = alg.nPoints

function vi(model, alg::ADQuadVI, q, θ_init; optimizer = TruncatedADAGrad())
    θ = copy(θ_init)
    optimize!(elbo, alg, q, model, θ; optimizer = optimizer)

    # If `q` is a mean-field approx we use the specialized `update` function
    if q isa Distribution
        return update(q, θ)
    else
        # Otherwise we assume it's a mapping θ → q
        return q(θ)
    end
end


function optimize(elbo::ELBO, alg::ADQuadVI, q, model, θ_init; optimizer = TruncatedADAGrad())
    θ = copy(θ_init)

    # `model` assumed to be callable z ↦ p(x, z)
    optimize!(elbo, alg, q, model, θ; optimizer = optimizer)

    return θ
end

# WITHOUT updating parameters inside ELBO
function (elbo::ELBO)(
    rng::AbstractRNG,
    alg::ADQuadVI,
    q::VariationalPosterior,
    logπ::Function,
    num_samples
)
    μ, σ² = get_diag_params(q)
    xs = [n * sqrt.(σ²) .+ μ for n in alg.nodes]
    # res = sum((x,w) -> _eval_logπ(q.transform, x)*w for (x,w) in zip(xs, alg.weights))
    res = eval_logπ(logπ, q.transform, xs[1]) * alg.weights[1]
    for i in 2:num_samples
        res += eval_logπ(logπ, q.transform, xs[i]) * alg.weights[i]
    end
    if q isa TransformedDistribution
        res += entropy(q.dist)
    else
        res += entropy(q)
    end

    return res
end

Distributions.params(d::TuringDenseMvNormal) = d.m, d.C
get_diag_params(q::TransformedDistribution) = get_diag_params(q.dist)
function get_diag_params(q::TuringDenseMvNormal)
    μ, L = params(q)
    return μ, vec(sum(L.L .* L.L', dims = 2))
end
get_diag_params(q::TuringDiagMvNormal) = params(q)
