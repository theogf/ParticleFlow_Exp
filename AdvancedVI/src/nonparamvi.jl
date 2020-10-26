using StatsFuns
using DistributionsAD
using Random: AbstractRNG, GLOBAL_RNG

struct MvMixtureModel{T,V<:AbstractVector{Vector{T}}} <:
       Distributions.ContinuousMultivariateDistribution
    dim::Int
    K::Int
    μ::V
    σ²::Vector{T}
    function MvMixtureModel(μ::V, σ²::AbstractVector{T}) where {T, V<:AbstractVector{Vector{T}}}
        new{T,V}(length(first(μ)), length(μ), μ, σ²)
    end
end

function update_q!(d::MvMixtureModel)
    d.μ .= vec(mean(d.x, dims = 2))
    d.Σ .= cov(d.x, dims = 2)
    nothing
end

Base.length(d::MvMixtureModel) = d.dim
nParticles(d::MvMixtureModel) = d.K
_realizep(d::MvMixureModel) = MixtureModel(MvNormal.(d.μ, d.σ²))

# Random._rand!(d::SteinDistribution, v::AbstractVector) = d.x
Base.eltype(::MvMixtureModel{T}) where {T} = T
function Distributions._rand!(
    rng::AbstractRNG,
    d::MvMixtureModel,
    x::AbstractVector,
)
    length(x) == d.dim || error("Wrong dimensions")
    x .= rand(rng, _realize_p(d))
end
function Distributions._rand!(
    rng::AbstractRNG,
    d::MvMixtureModel,
    x::AbstractMatrix,
)
    nDim, nPoints = size(x)
    nDim == d.dim || error("Wrong dimensions")
    x .= rand(rng, _realize_p(d), nPoints)
end
Distributions.mean(d::MvMixtureModel) = mean(d.μ)
_var(d::MixtureModel) = mean(d.σ²) + mean(d.)
Distributions.cov(d::MvMixtureModel) = mean(d.σ²) * I(length(d)) + mean((d.μ .- mean(d)) .* (d.μ .- mean(d))')
Distributions.var(d::MvMixtureModel) = mean(d.σ²) * ones(length(d)) + mean(abs2.(d.μ .- mean(d)))
Distributions.entropy(d::MvMixtureModel) = NaN

q_to_vec(q::MvMixtureModel) = vcat(q.μ..., q.σ²)
vec_to_q(q::MvMixtureModel, θ::AbstractVector) = MvMixtureModel([θ[((i-1) * q.dim + 1):(i * q.dim)] for i in 1:q.K], θ[(q.K * q.dim + 1):end])


const AllMvMixtureModel =
    Union{MvMixtureModel,TransformedDistribution{<:MvMixtureModel}}

"""
    PFlowVI(n_particles = 100, max_iters = 1000)

Gaussian Particle Flow Inference (PFlowVI) for a given model.
"""
struct NonParamVI{AD} <: VariationalInference{AD}
    max_iters::Int        # maximum number of gradient steps used in optimization
end

# params(alg::SteinVI) = nothing;#params(alg.kernel)

NonParamVI(args...) = NonParamVI{ADBackend()}(args...)
NonParamVI() = NonParamVI(100)

alg_str(::NonParamVI) = "NonParamVI"

function vi(
    logπ::Function,
    alg::NonParamVI,
    q::MvMixtureModel;
    optimizer = TruncatedADAGrad(),
    callback = nothing,
    hyperparams = nothing,
    hp_optimizer = nothing,
)
    DEBUG && @debug "Optimizing $(alg_str(alg))..."
    # Initial parameters for mean-field approx
    # Optimize
    optimize!(
        elbo,
        alg,
        transformed(q, Identity{1}()),
        logπ,
        q_to_vec(q);
        optimizer = optimizer,
        callback = callback,
        hyperparams = hyperparams,
        hp_optimizer = hp_optimizer,
    )

    # Return updated `Distribution`
    return q
end

function vi(
    logπ::Function,
    alg::NonParamVI,
    q::TransformedDistribution{<:MvMixtureModel};
    optimizer = TruncatedADAGrad(),
    callback = nothing,
    hyperparams = nothing,
    hp_optimizer = nothing,
)
    DEBUG && @debug "Optimizing $(alg_str(alg))..."
    # Initial parameters for mean-field approx
    # Optimize
    optimize!(
        elbo,
        alg,
        q,
        logπ,
        q_to_vec(q);
        optimizer = optimizer,
        callback = callback,
        hyperparams = nothing,
        hp_optimizer = nothing,
    )

    # Return updated `Distribution`
    return q
end

function grad!(
    vo,
    alg::NonParamVI{<:ForwardDiffAD},
    q,
    logπ,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    args...
)
    f(x) = sum(mapslices(
        z -> phi(logπ, q, z),
        x,
        dims = 1,
    ))
    chunk_size = getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(q.dist.x), chunk_size))
    config = ForwardDiff.GradientConfig(f, q.dist.x, chunk)
    ForwardDiff.gradient!(out, f, q.dist.x, config)
end

phi(logπ, q, x) = -eval_logπ(logπ, q, x)

qₙ(q, n) = sum(pdf(MvNormal(q.μ[j], (q.σ²[n] + q.σ²[n])), q.μ[n]) for i in 1:q.K)
function elbo(logπ, q)
    sum(logπ(q.μ[i])) + q.σ²[i] * trH(logπ, q.μ[i]) + log(qₙ(q, i)) for i in 1:q.K) / q.K
end

function elbo_1(logπ, q)
    sum(logπ(q.μ[i]) + log(qₙ(q, i)) for i in 1:q.K) / q.K
end

function elbo_2(logπ, q)
    sum(q.σ²[i] * trH(logπ, q.μ[i]) + log(qₙ(q, i)) for i in 1:q.K) / q.K
end

function optimize!(
    vo,
    alg::NonParamVI,
    q::SampMvNormal,
    logπ,
    θ::AbstractVector{<:Real};
    optimizer = TruncatedADAGrad(),
    callback = nothing,
    hyperparams = nothing,
    hp_optimizer = nothing
)
    alg_name = alg_str(alg)
    samples_per_step = nSamples(alg)
    max_iters = alg.max_iters

    optimizer = if Base.isiterable(typeof(optimizer))
        length(optimizer) == 2 || error("Optimizer should be of size 2 only")
        optimizer
    else
        fill(optimizer, 2)
    end

    diff_result = DiffResults.GradientResult(θ)

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0)
    else
        0
    end

    time_elapsed = @elapsed while (i < max_iters) # & converged

        _logπ = if !isnothing(hyperparams)
            logπ(hyperparams)
        else
            logπ
        end

        grad!(vo, alg, q, _logπ, θ, diff_result, samples_per_step)

        Δ = DiffResults.gradient(diff_result)

        gradient(θ)

        # apply update rule
        Δ₁ = apply!(optimizer[1], q.dist.μ, Δ₁)
        Δ₂ = apply!(optimizer[2], q.dist.x, Δ₂)
        @. q.dist.x = q.dist.x - Δ₁ - Δ₂
        update_q!(q.dist)

        if !isnothing(hyperparams) && !isnothing(hp_optimizer)
            Δ = hp_grad(vo, alg, q, logπ, hyperparams)
            Δ = apply!(hp_optimizer, hyperparams, Δ)
            hyperparams .+= Δ
        end

        if !isnothing(callback)
            callback(i, q, hyperparams)
        end
        AdvancedVI.DEBUG && @debug "Step $i" Δ
        PROGRESS[] && (ProgressMeter.next!(prog))

        i += 1
    end

    return q
end

function (elbo::ELBO)(
    rng::AbstractRNG,
    alg::PFlowVI,
    q::TransformedDistribution{<:MvMixtureModel},
    logπ::Function
)

    res = sum(mapslices(x -> -phi(logπ, q, x), q.dist.x, dims = 1))

    if q isa TransformedDistribution
        res += entropy(q.dist)
    else
        res += entropy(q)
    end
    return res
end
