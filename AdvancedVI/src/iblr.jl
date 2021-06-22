"""
    IBLR(n_particles = 100, max_iters = 1000)

iBayes Learning Rule (IBLR) for a given model.
Can only work on the following distributions:
 - `Precision`
 - `MFMvNormal`
"""
struct IBLR{AD} <: GVA{AD}
    max_iters::Int        # maximum number of gradient steps used in optimization
    nSamples::Int   # Number of samples per expectation
    hess_comp::Symbol
end

# params(alg::SteinVI) = nothing;#params(alg.kernel)

IBLR(args...) = IBLR{ADBackend()}(args...)
IBLR() = IBLR(100, 100, :hess)

alg_str(::IBLR) = "IBLR"

function hessian!( # Does not work currently... 
    alg::IBLR{<:ForwardDiffAD},
    q,
    logπ,
    x,
    out::AbstractVector{<:DiffResults.MutableDiffResult},
    args...,
)
    f(x) = phi(logπ, q, x)
    chunk_size = getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(x[:, 1]), chunk_size))
    config = ForwardDiff.HessianConfig.(f, eachcol(x), Ref(chunk))
    ForwardDiff.hessian!.(out, f, eachcol(x), config)
end

function optimize!(
    vo,
    alg::IBLR,
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

    z = zeros(dim(q.dist), samples_per_step) # Storage for samples
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
    G = similar(q.dist.S)
    gS = similar(q.dist.S)
    # optimizer isa Descent || error("IBLR only work with std. grad. descent")
    
    time_elapsed = @elapsed while (i < max_iters) # & converged


        _logπ = if !isnothing(hyperparams)
            logπ(hyperparams)
        else
            logπ
        end
        
        Distributions.randn!(z)

        reparametrize!(x, q.dist, z)

        grad!(alg, q, _logπ, x, diff_result, samples_per_step)
        # hessian!(vo, alg, q, _logπ, x, hess_results, samples_per_step)
        Δ = DiffResults.gradient(diff_result)
        

        update_dist!(q.dist, alg, _logπ, Δ, Δμ, G, gS, x, z, optimizer)
        
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


function update_dist!(d::PrecisionMvNormal, alg::IBLR, logπ, Δ, Δμ, G, gS, x, z, opt)
    Δt = opt.eta
    if alg.hess_comp == :hess
        gS .= mean(x->Zygote.hessian(z->phi(logπ, d, z), x), eachcol(x))
    elseif alg.hess_comp == :rep
        gS .= z * Δ' / nSamples(alg)
        gS .= 0.5 * (gS + gS')
    end

    G .= d.S - gS
    Δμ .= d.S \ vec(mean(Δ, dims=2))
    d.μ .-= Δt * Δμ
    d.S .= Symmetric((1 - Δt) * d.S + Δt * gS + 0.5 * Δt^2 * G * (d.S \ G))
end

function update_dist!(d::DiagPrecisionMvNormal, alg::IBLR, logπ, Δ, Δμ, G, gS, x, z, opt)
    Δt = opt.eta
    if alg.hess_comp == :hess
        gS .= mean(diag.(ForwardDiff.hessian.(z->phi(logπ, d, z), eachcol(x))))
    elseif alg.hess_comp == :rep
        gS .= diag_ABt(z, Δ) / nSamples(alg)
    end
    G .= d.S - gS
    Δμ .= d.S .\ vec(mean(Δ, dims=2))
    d.μ .-= Δt * Δμ
    d.S .= (1 - Δt) * d.S + Δt * gS + 0.5 * Δt^2 * G .* (d.S .\ G)
end


mutable struct ADAM_IBLR{R, V}
    r::R # Decaying rates
    η::Int # General learning rate
    k::Int # Time
    m::V # Momentum for mean
    S::V # Momentum for precision
end

function update_dist!(d::DiagPrecisionMvNormal, alg::IBLR, logπ, Δ, Δμ, G, gS, x, z, opt::ADAM_IBLR)
    r = opt.r
    k = opt.k
    η = opt.η

    g = vec(mean(Δ, dims=2))
    gS .= diag_ABt(d.S .* (x .- d.μ), Δ) / nSamples(alg)
    G .= d.S - gS
    
    opt.m .= r[1] * opt.m + (1 - r[1]) * g # Update of the momentum
    d.μ .= d.μ - η * (1 - r[2]^k) / (1 - r[1]^k) * d.S .\ ((1 - r[1]) * g + r[1] * opt.m)
    d.S .= 0.5 * (d.S + (d.S + (1 - r[2]) * h).^2 ./ d.S)

    opt.k += 1
end
