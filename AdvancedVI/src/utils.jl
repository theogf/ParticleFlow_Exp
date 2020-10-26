update(td::TransformedDistribution, θ...) = transformed(update(td.dist, θ...), td.transform)

update(d::TuringDiagMvNormal, μ, σ) = TuringDiagMvNormal(μ, σ)
function update(td::Union{<:TransformedDistribution{D}, D}, θ::AbstractArray) where {D<:TuringDiagMvNormal}
    μ, σ = θ[1:length(td)], exp.(θ[length(td) + 1:end])
    return update(td, μ, σ)
end

function update(td::TuringDenseMvNormal{<:AbstractVector, <:Cholesky{<:Real, <:BlockDiagonal}}, θ::AbstractArray)
    μ = θ[1:length(td)]
    sizes = vcat(0, first.(blocksizes(td.C.U)))
    ids = cumsum((sizes .* (sizes .+ 1)) .÷ 2)
    Σ = θ[(length(td)+1):end]
    Σs = [make_triangular(Σ[(ids[i-1]+1):ids[i]], sizes[i]) for i in 2:length(ids)]
    return update(td, μ, BlockDiagonal(Σs))
end

update(d::TuringDenseMvNormal, μ, L) = TuringDenseMvNormal(μ, L * L' + 1e-5I)

function update(td::TuringDenseMvNormal, θ::AbstractArray)
    μ, L = θ[1:length(td)], make_triangular(θ[length(td) + 1:end], length(td))
    return update(td, μ, L)
end

function make_triangular(x, D)
    [i >= j ? x[div(j * (j - 1), 2)+i] : zero(eltype(x)) for i = 1:D, j = 1:D]
end

function eval_logπ(logπ, q::TransformedDistribution, x)
    z, logjac = forward(q.transform, x)
    return logπ(z) + logjac
end

function eval_logπ(logπ, q::Distribution, x)
    return logπ(x)
end

function hp_grad(vo, alg, q, logπ, hyperparameters, args...)
    ForwardDiff.gradient(x -> vo(alg, q, logπ(x), args...), hyperparameters)
end

function LinearAlgebra.:\(A::BlockDiagonal, B::AbstractVecOrMat)
    i = 1
    c = similar(B)
    for a in blocks(A)
        d = size(a, 1)
        c[i:i+d-1, :] = a \ B[i:i+d-1, :]
        i += d
    end
    return c
end
