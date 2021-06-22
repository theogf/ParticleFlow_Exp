module AdvancedVI

using Random: AbstractRNG, GLOBAL_RNG, Random

using Distributions, DistributionsAD, Bijectors
using FastGaussQuadrature
using BlockDiagonals
using DocStringExtensions
using StatsFuns
using ProgressMeter, LinearAlgebra

using KernelFunctions, Distances

using ForwardDiff
using Tracker
using Functors

export
    vi,
    ADVI,
    ADQuadVI,
    ELBO,
    elbo,
    TruncatedADAGrad,
    DecayedADAGrad,
    VariationalInference,
    SVGD, IBLR, DSVI, FCS,
    GaussFlow, GaussPFlow,
    EmpiricalDistribution,
    SamplesMvNormal, BlockMFSamplesMvNormal, MFSamplesMvNormal,
    LowRankMvNormal, BlockMFLowRankMvNormal, MFMvNormal, FCSMvNormal,
    PrecisionMvNormal, CholMvNormal

@functor TuringDenseMvNormal
@functor BlockDiagonal

const PROGRESS = Ref(true)
function turnprogress(switch::Bool)
    @info("[AdvancedVI]: global PROGRESS is set as $switch")
    PROGRESS[] = switch
end

const DEBUG = Bool(parse(Int, get(ENV, "DEBUG_ADVANCEDVI", "0")))

include("ad.jl")

using Requires
function __init__()
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        apply!(o, x, Δ) = Flux.Optimise.apply!(o, x, Δ)
        Flux.Optimise.apply!(o::TruncatedADAGrad, x, Δ) = apply!(o, x, Δ)
        Flux.Optimise.apply!(o::DecayedADAGrad, x, Δ) = apply!(o, x, Δ)
    end
    @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
        include("compat/zygote.jl")
    end
    @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
        include("compat/reversediff.jl")
    end
    @require Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0" begin
        include("turingmodels.jl")
    end

end

abstract type VariationalInference{AD} end

getchunksize(::Type{<:VariationalInference{AD}}) where AD = getchunksize(AD)
getADtype(::VariationalInference{AD}) where AD = AD

abstract type VariationalObjective end

const VariationalPosterior = Distribution{Multivariate, Continuous}

nSamples(::VariationalInference) = nothing
alg_str(alg::VariationalInference) = "Undefined, please define `alg_str(alg)`"

"""
    grad!(vo, alg::VariationalInference, q, model::Model, θ, out, args...)

Computes the gradients used in `optimize!`. Default implementation is provided for
`VariationalInference{AD}` where `AD` is either `ForwardDiffAD` or `TrackerAD`.
This implicitly also gives a default implementation of `optimize!`.

Variance reduction techniques, e.g. control variates, should be implemented in this function.
"""
function grad! end

"""
    vi(model, alg::VariationalInference)
    vi(model, alg::VariationalInference, q::VariationalPosterior)
    vi(model, alg::VariationalInference, getq::Function, θ::AbstractArray)

Constructs the variational posterior from the `model` and performs the optimization
following the configuration of the given `VariationalInference` instance.

# Arguments
- `model`: `Turing.Model` or `Function` z ↦ log p(x, z) where `x` denotes the observations
- `alg`: the VI algorithm used
- `q`: a `VariationalPosterior` for which it is assumed a specialized implementation of the variational objective used exists.
- `getq`: function taking parameters `θ` as input and returns a `VariationalPosterior`
- `θ`: only required if `getq` is used, in which case it is the initial parameters for the variational posterior
"""
function vi end

function update end

# default implementations
function grad!(
    vo,
    alg::VariationalInference{<:ForwardDiffAD},
    q,
    model,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    args...
)
    f(θ_) = if (q isa Distribution)
        - vo(alg, update(q, θ_), model, args...)
    else
        - vo(alg, q(θ_), model, args...)
    end

    chunk_size = getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(θ), chunk_size))
    config = ForwardDiff.GradientConfig(f, θ, chunk)
    ForwardDiff.gradient!(out, f, θ, config)
end

function grad!(
    vo,
    alg::VariationalInference{<:TrackerAD},
    q,
    model,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    args...
)
    θ_tracked = Tracker.param(θ)
    y = if (q isa Distribution)
        - vo(alg, update(q, θ_tracked), model, args...)
    else
        - vo(alg, q(θ_tracked), model, args...)
    end
    Tracker.back!(y, 1.0)

    DiffResults.value!(out, Tracker.data(y))
    DiffResults.gradient!(out, Tracker.grad(θ_tracked))
end


"""
    optimize!(vo, alg::VariationalInference{AD}, q::VariationalPosterior, model::Model, θ; optimizer = TruncatedADAGrad())

Iteratively updates parameters by calling `grad!` and using the given `optimizer` to compute
the steps.
"""
function optimize!(
    vo,
    alg::VariationalInference,
    q,
    model,
    θ::AbstractVector{<:Real};
    optimizer = TruncatedADAGrad(),
    callback = nothing,
    hyperparams = nothing,
    hp_optimizer = nothing
)
    # TODO: should we always assume `samples_per_step` and `max_iters` for all algos?
    alg_name = alg_str(alg)
    samples_per_step = nSamples(alg)
    max_iters = alg.max_iters

    num_params = length(θ)

    # TODO: really need a better way to warn the user about potentially
    # not using the correct accumulator
    if (optimizer isa TruncatedADAGrad) && (θ ∉ keys(optimizer.acc))
        # this message should only occurr once in the optimization process
        @info "[$alg_name] Should only be seen once: optimizer created for θ" objectid(θ)
    end

    diff_result = DiffResults.GradientResult(θ)

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0)
    else
        0
    end

    # add criterion? A running mean maybe?
    # time_elapsed = @elapsed
    while (i < max_iters) # & converged
        logπ = if isnothing(hyperparams)
            model
        else
            model(hyperparams)
        end

        grad!(vo, alg, q, logπ, θ, diff_result, samples_per_step)

        # apply update rule
        Δ = DiffResults.gradient(diff_result)
        Δ = apply!(optimizer, θ, Δ)
        @. θ = θ - Δ

        AdvancedVI.DEBUG && @debug "Step $i" Δ DiffResults.value(diff_result)
        PROGRESS[] && (ProgressMeter.next!(prog))
        if !isnothing(hyperparams) && !isnothing(hp_optimizer)
            Δ = hp_grad(vo, alg, q, model, hyperparams, samples_per_step)
            Δ = apply!(hp_optimizer, hyperparams, Δ)
            hyperparams .+= Δ
        end
        if !isnothing(callback)
            callback(i, update(q, θ), hyperparams)
        end
        i += 1
    end

    return θ
end

# objectives
include("objectives.jl")

# optimisers
include("optimisers.jl")

# special distributions
include("dists.jl")

# VI algorithms
include("advi.jl")
include("gva.jl")
include("adquadvi.jl")
include("svgd.jl")
include("utils.jl")



end # module
