"""
    GaussPFlow(n_particles = 100, max_iters = 1000)

Gaussian Particle Flow Inference (GaussPFlow) for a given model.
"""
struct GaussPFlow{AD} <: GVA{AD}
    max_iters::Int        # maximum number of gradient steps used in optimization
    precondΔ₁::Bool # Precondition the first gradient (mean)
    precondΔ₂::Bool # Precondition the second gradient (cov)
end

# params(alg::SteinVI) = nothing;#params(alg.kernel)

GaussPFlow(args...) = GaussPFlow{ADBackend()}(args...)
GaussPFlow() = GaussPFlow(100, true, false)

alg_str(::GaussPFlow) = "GaussPFlow"

phi(logπ, q, x) = -eval_logπ(logπ, q, x)

function optimize!(
    vo,
    alg::GaussPFlow,
    q::SampMvNormal,
    logπ;
    optimizer = TruncatedADAGrad(),
    callback = nothing,
    hyperparams = nothing,
    hp_optimizer = nothing,
)
    alg_name = alg_str(alg)
    max_iters = alg.max_iters

    optimizer = if optimizer isa AbstractVector #Base.isiterable(typeof(optimizer))
        length(optimizer) == 2 || error("Optimizer should be of size 2 only")
        optimizer
    else
        fill(optimizer, 2)
    end

    diff_result = DiffResults.GradientResult(q.dist.x)

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0)
    else
        0
    end
    Δ₁ = similar(q.dist.μ)
    Δ₂ = similar(q.dist.x)
    time_elapsed = @elapsed while (i < max_iters) # & converged

        _logπ = if !isnothing(hyperparams)
            logπ(hyperparams)
        else
            logπ
        end

        grad!(alg, q, _logπ, diff_result)

        Δ = DiffResults.gradient(diff_result)

        Δ₁ .= if alg.precondΔ₁
            cov(q.dist) * vec(mean(Δ, dims = 2))
        else
            vec(mean(Δ, dims = 2))
        end
        shift_x = q.dist.x .- q.dist.μ
        compute_cov_part!(Δ₂, q.dist, shift_x, Δ, alg)

        # apply update rule
        Δ₁ .= apply!(optimizer[1], q.dist.μ, Δ₁)
        Δ₂ .= apply!(optimizer[2], q.dist.x, Δ₂)
        @. q.dist.x = q.dist.x - Δ₁ - Δ₂
        update_q!(q.dist)

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

function compute_cov_part!(
    Δ₂::AbstractMatrix,
    q::SamplesMvNormal,
    x::AbstractMatrix,
    Δ::AbstractMatrix,
    alg::GaussPFlow,
)
    Δ₂ .= x
    if alg.precondΔ₂
        A = Δ * x' / nParticles(q) - I
        B = inv(q.Σ) # Approximation hessian
        # B = Δ * Δ' # Gauss-Newton approximation
        cond = tr(A' * A) / (tr(A^2) + tr(A' * B * A * q.Σ))
        lmul!(cond * A, Δ₂)
    else
        if nParticles(q) < q.dim
            mul!(Δ₂, Δ, x' * x, Float32(inv(nParticles(q))), -1.0f0)
            # If N >> D it's more efficient to compute ϕ xᵀ first
        else
            mul!(Δ₂, Δ * x', x, Float32(inv(nParticles(q))), -1.0f0)
        end
    end
end

function compute_cov_part!(
    Δ₂::AbstractMatrix,
    q::BlockMFSamplesMvNormal,
    x::AbstractMatrix,
    Δ::AbstractMatrix,
    ::GaussPFlow,
)
    Δ₂ .= x
    for i = 1:q.K
        xview = x[(q.id[i]+1):q.id[i+1], :]
        # Proceed to the operation :
        # (ψ - I) * x == (1/ N (∑ ϕᵢxᵀᵢ) - I) * x == (1/N ϕ xᵀ - I) * x
        # It is done via mul!(C, A, B, α, β) : C = αAB + βC
        # If D << N it's more efficient to compute xᵀ x first
        if q.n_particles < q.id[i+1] - q.id[i]
            mul!(
                Δ₂[(q.id[i]+1):q.id[i+1], :],
                Δ[(q.id[i]+1):q.id[i+1], :],
                xview' * xview,
                Float32(inv(nParticles(q))),
                -1.0f0,
            )
            # If N >> D it's more efficient to compute ϕ xᵀ first
        else
            mul!(
                Δ₂[(q.id[i]+1):q.id[i+1], :],
                Δ[(q.id[i]+1):q.id[i+1], :] * xview',
                xview,
                Float32(inv(nParticles(q))),
                -1.0f0,
            )
        end
    end
end

function compute_cov_part!(
    Δ₂::AbstractMatrix,
    q::MFSamplesMvNormal,
    x::AbstractMatrix,
    Δ::AbstractMatrix,
    ::GaussPFlow,
)
    ## Multiply g by x, and take the diagonal, then multiply by
    Δ₂ .= (sum((Δ .*= x), dims=2) / nParticles(q) .- 1) .* x 
end

function (elbo::ELBO)(
    ::AbstractRNG,
    ::GaussPFlow,
    q::TransformedDistribution{<:SamplesMvNormal},
    logπ::Function,
)
    res = sum(mapslices(x -> -phi(logπ, q, x), q.dist.x, dims = 1))
    res += entropy(q.dist)
    return res
end
