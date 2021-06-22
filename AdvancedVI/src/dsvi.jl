"""
    DSVI(n_particles = 100, max_iters = 1000)

Doubly Stochastic Variational Inference (DSVI) for a given model.
Can only work on the following distributions:
 - `CholMvNormal`
 - `MFMvNormal`
"""
struct DSVI{AD,RNG,F} <: GVA{AD}
    rng::RNG
    device::F
    max_iters::Int        # maximum number of gradient steps used in optimization
    nSamples::Int   # Number of samples per expectation
end

# params(alg::SteinVI) = nothing;#params(alg.kernel)

DSVI(rng::RNG, device::F, args...) where {RNG, F} = DSVI{ADBackend(),RNG,F}(rng, device, args...)
DSVI() = DSVI(Random.GLOBAL_RNG, identity, 100, 10)

alg_str(::DSVI) = "DSVI"

function optimize!(
    vo,
    alg::DSVI,
    q::Bijectors.TransformedDistribution,
    logπ;
    optimizer = TruncatedADAGrad(),
    callback = nothing,
    hyperparams = nothing,
    hp_optimizer = nothing,
)
    alg_name = alg_str(alg)
    samples_per_step = nSamples(alg)
    max_iters = alg.max_iters

    optimizer = if optimizer isa AbstractVector #Base.isiterable(typeof(optimizer))
        length(optimizer) == 2 || error("Optimizer should be of size 2 only")
        optimizer
    else
        fill(optimizer, 2)
    end

    x₀ = alg.device(zeros(dim(q.dist), samples_per_step)) # Storage for raw samples
    x = alg.device(zeros(dim(q.dist), samples_per_step)) # Storage for samples
    diff_result = DiffResults.GradientResult(x)

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0)
    else
        0
    end
    Δμ = similar(q.dist.μ)
    ΔΓ = similar(q.dist.Γ)
    time_elapsed = @elapsed while (i < max_iters) # & converged

        _logπ = if !isnothing(hyperparams)
            logπ(hyperparams)
        else
            logπ
        end
        
        Random.randn!(alg.rng, x₀)
        reparametrize!(x, q.dist, x₀)

        grad!(alg, q, _logπ, x, diff_result, samples_per_step)

        Δ = DiffResults.gradient(diff_result)
        
        Δμ .= apply!(optimizer[1], q.dist.μ, vec(mean(Δ, dims = 2)))
        ΔΓ .= typeof(q.dist.Γ)(apply!(optimizer[2],
                                q.dist.Γ isa LowerTriangular ? q.dist.Γ.data : q.dist.Γ,
                                updateΓ(Δ, x₀, q.dist.Γ))
                            )
        # apply update rule
        q.dist.μ .-= Δμ
        q.dist.Γ .-= ΔΓ

        if !isnothing(hyperparams) && !isnothing(hp_optimizer)
            Δ = hp_grad(vo, alg, q, logπ, hyperparams)
            Δ = apply!(hp_optimizer, hyperparams, Δ)
            hyperparams .+= Δ
        end
        AdvancedVI.DEBUG && @debug "Step $i" Δ
        PROGRESS[] && (ProgressMeter.next!(prog))

        if !isnothing(callback)
            callback(i, q, hyperparams)
        end
        i += 1
    end

    return q
end

function updateΓ(Δ::AbstractMatrix, z::AbstractMatrix, Γ::AbstractVector)
    vec(mean(Δ .* z, dims=2)) - inv.(Γ)
end

function updateΓ(Δ::AbstractMatrix, z::AbstractMatrix, Γ::LowerTriangular)
    LowerTriangular(Δ * z' / size(z, 2)) - inv(Diagonal(Γ))
end