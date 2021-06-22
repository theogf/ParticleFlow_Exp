using .ReverseDiff: compile, GradientTape
using .ReverseDiff.DiffResults: GradientResult

struct ReverseDiffAD{cache} <: ADBackend end
const RDCache = Ref(false)
setcache(b::Bool) = RDCache[] = b
getcache() = RDCache[]
ADBackend(::Val{:reversediff}) = ReverseDiffAD{getcache()}
function setadbackend(::Val{:reversediff})
    ADBACKEND[] = :reversediff
end

tape(f, x) = GradientTape(f, x)
function taperesult(f, x)
    return tape(f, x), GradientResult(x)
end

export ReverseDiffAD

function AdvancedVI.grad!(
    vo,
    alg::VariationalInference{<:AdvancedVI.ReverseDiffAD{false}},
    q,
    model,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    args...
)
    f(θ) = if (q isa Distribution)
        - vo(alg, update(q, θ), model, args...)
    else
        - vo(alg, q(θ), model, args...)
    end
    tp = AdvancedVI.tape(f, θ)
    ReverseDiff.gradient!(out, tp, θ)
    return out
end

function grad!(
    ::Union{GaussPFlow{<:ReverseDiffAD},SVGD{<:ReverseDiffAD}},
    q,
    logπ,
    out::DiffResults.MutableDiffResult,
    args...
)
    f(x) = sum(eachcol(x)) do z
        phi(logπ, q, z)
    end
    tp = AdvancedVI.tape(f, q.dist.x)
    ReverseDiff.gradient!(out, tp, q.dist.x)
    return out
end

function grad!(
    ::GVA{<:ReverseDiffAD},
    q,
    logπ,
    x,
    out::DiffResults.MutableDiffResult,
    args...
)
    f(x) = sum(map(
        z -> phi(logπ, q, z),
        eachcol(x),
    ))
    tp = AdvancedVI.tape(f, x)
    ReverseDiff.gradient!(out, tp, x)
    return out
end