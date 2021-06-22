struct ZygoteAD <: ADBackend end
ADBackend(::Val{:zygote}) = ZygoteAD
function setadbackend(::Val{:zygote})
    ADBACKEND[] = :zygote
end

import .Zygote

export ZygoteAD

function AdvancedVI.grad!(
    vo,
    alg::VariationalInference{<:AdvancedVI.ZygoteAD},
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
    y, back = Zygote.pullback(f, θ)
    dy = first(back(1.0))
    DiffResults.value!(out, y)
    DiffResults.gradient!(out, dy)
    return out
end

function grad!(
    ::Union{GaussPFlow{<:AdvancedVI.ZygoteAD},SVGD{<:AdvancedVI.ZygoteAD}},
    q,
    logπ,
    out::DiffResults.MutableDiffResult,
    args...
)
    f(x) = sum(eachcol(x)) do z
        phi(logπ, q, z)
    end
    val, back = Zygote.pullback(f, q.dist.x)
    dy = first(back(1.0))
    DiffResults.value!(out, val)
    DiffResults.gradient!(out, dy)
    return out
end

function grad!(
    ::GVA{<:AdvancedVI.ZygoteAD},
    q,
    logπ,
    x,
    out::DiffResults.MutableDiffResult,
    args...
)
    f(x) = sum(eachcol(x)) do z
        phi(logπ, q, z)
    end
    val, back = Zygote.pullback(f, x)
    dy = first(back(1.0))
    DiffResults.value!(out, val)
    DiffResults.gradient!(out, dy)
    return out
end