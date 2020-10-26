struct MeanFieldQ{A} <: Distributions.ContinuousMultivariateDistribution
    dists::A
    dims::Vector{Int}
end

function MeanFieldQ(dists::T) where {T<:AbstractVector}
    return MeanFieldQ{T}(dists, length.(dists))
end

Base.length(d::MeanFieldQ) = sum(d.dims)

function Distributions._rand!(
    rng::AbstractRNG,
    d::MeanFieldQ,
    x::AbstractVector,
)
    nDim = length(x)
    @assert nDim == length(d) "Wrong dimensions"
    x .= foldl(vcat, rand(_d) for d in d)
end

function Distributions._rand!(
    rng::AbstractRNG,
    d::MeanFieldQ,
    x::AbstractMatrix,
)
    nDim, nPoints = size(x)
    @assert nDim == d.dim "Wrong dimensions"
    x .= d.x[rand(rng, 1:d.n_particles, nPoints), :]'
end

Distributions.mean(d::MeanFieldQ) = vcat(mean.(d.dists))
Distributions.cov(d::MeanFieldQ) = BlockDiagonal(cov.(d.dists))
Distributions.var(d::MeanFieldQ) = vcat(var.(d.dists))
Distributions.entropy(d::MeanFieldQ) = sum(entropy.(d.dists))


function vi(
    logπ::Function,
    alg::AbstractVector{<:VariationalInference},
    q::MeanFieldQ;
    optimizer = TruncatedADAGrad(),
    callback = nothing,
    hyperparams = nothing,
    hp_optimizer = nothing,
)
    DEBUG && @debug "Optimizing mean_field with $(alg_str.(alg))..."
    # Initial parameters for mean-field approx
    # Optimize
    optimize!(
        elbo,
        alg,
        transformed(q, Identity{1}()),
        logπ,
        [0.0];
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
    alg::AbstractVector{<:VariationalInference},
    q::TransformedDistribution{<:MeanFieldQ};
    optimizer = TruncatedADAGrad(),
    callback = nothing,
    hyperparams = nothing,
    hp_optimizer = nothing,
)
    DEBUG && @debug "Optimizing mean-field with $(alg_str.(alg))..."
    # Initial parameters for mean-field approx
    # Optimize
    optimize!(
        elbo,
        alg,
        q,
        logπ,
        [0.0];
        optimizer = optimizer,
        callback = callback,
        hyperparams = nothing,
        hp_optimizer = nothing,
    )

    # Return updated `Distribution`
    return q
end

function optimize!(
    vo,
    algs::AbstractVector{<:VariationalInference},
    q::TransformedDistribution{<:MeanFieldQ},
    logπ,
    θ::AbstractVector{<:Real};
    optimizer = TruncatedADAGrad(),
    callback = nothing,
    hyperparams = nothing,
    hp_optimizer = nothing
)
    alg_name = alg_str.(alg)
    max_iters = nSamples.(alg)

    # optimizer = if Base.isiterable(typeof(optimizer))
    #     length(optimizer) == 2 || error("Optimizer should be of size 2 only")
    #     optimizer
    # else
    #     fill(optimizer, 2)
    # end

    # diff_result = DiffResults.GradientResult(θ)

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

        gs = grads!.(ELBO, alg, q.dists, )

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
    q::TransformedDistribution{<:MeanFieldQ},
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
