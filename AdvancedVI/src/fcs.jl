"""
    FCS(n_particles = 100, max_iters = 100)

Factorized Covariance Structure, Ong 2017
Can only work with the following distributions:
- `FCSMvNormal`
"""
struct FCS{AD} <: GVA{AD}
    max_iters::Int        # maximum number of gradient steps used in optimization
    nSamples::Int   # Number of samples per expectation
end

FCS(args...) = FCS{ADBackend()}(args...)
FCS() = FCS(100, 100)

alg_str(::FCS) = "FCS"

function optimize!(
    vo,
    alg::FCS,
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

    z = zeros(size(q.dist.Γ, 2), samples_per_step) # Storage for samples
    ϵ = zeros(dim(q.dist), samples_per_step) # Storage for samples
    x = zeros(dim(q.dist), samples_per_step) # Storage for samples
    diff_result = DiffResults.GradientResult(x)
    # hess_results = DiffResults.HessianResult.(eachcol(x)) 

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0)
    else
        0
    end
    Δμ = similar(q.dist.μ)
    
    time_elapsed = @elapsed while (i < max_iters) # & converged


        _logπ = if !isnothing(hyperparams)
            logπ(hyperparams)
        else
            logπ
        end
        
        Distributions.randn!(ϵ)
        Distributions.randn!(z)

        reparametrize!(x, q.dist, z, ϵ)

        grad!(alg, q, _logπ, x, diff_result, samples_per_step)
        # hessian!(vo, alg, q, _logπ, x, hess_results, samples_per_step)
        Δ = DiffResults.gradient(diff_result)
        
        A = computeA(q.dist.Γ, q.dist.D)
        Δμ .= apply!(optimizer, q.dist.μ, vec(mean(-Δ, dims=2)))
        ΔΓ = apply!(optimizer, q.dist.Γ, gradΓ(-Δ, ϵ, z, q.dist.Γ, q.dist.D, A))
        ΔD = apply!(optimizer, q.dist.D, gradD(-Δ, ϵ, z, q.dist.Γ, q.dist.D, A))
        q.dist.μ .+= Δμ
        q.dist.Γ .+= ΔΓ
        q.dist.D .+= ΔD

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

function computeA(Γ::AbstractMatrix, D::AbstractVector)
    Dinv = Diagonal(inv.(D.^2))
    return Dinv - Dinv * Γ * inv(I + Γ' * Dinv * Γ) * Γ' * Dinv
end

function gradΓ(g, ϵ, z, Γ, D, A)
    return (g * z' + A * (Γ * z + D .* ϵ) * z') / size(z, 2)
end

function gradD(g, ϵ, z, Γ, D, A)
    return vec(mean(g .* ϵ, dims =2)) + diag(A * (Γ * z + D .* ϵ) * ϵ') / size(ϵ, 2)
end

# function ELBO(d::FCS, logπ; nSamples::Int=nSamples(d))
#     A = computeA(q.dist.Γ, q.dist.D)
#     sum(logπ, eachcol(rand(d, nSamples))) / nSamples + 0.5 * logdet(cov(d))
#     # sum(x->logπ(x) + 0.5 * invquad(cov(d), x - mean(d)) , eachcol(rand(d, nSamples))) / nSamples + 0.5 * logdet(cov(d))
# end