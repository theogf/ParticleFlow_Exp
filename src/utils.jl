using ChainRulesCore

invquad(A::AbstractMatrix, x::AbstractVecOrMat) = dot(x, A \ x)
XXt(X::AbstractVecOrMat) = X * X'
function gradcol(::Val{:ForwardDiff}, f, X)
    ForwardDiff.gradient(x->sum(f, eachcol(x)), X)
end

function gradcol(::Val{:Zygote}, f::Function, X::AbstractMatrix)
    first(Zygote.gradient(x->sum(f, eachcol(x)), X))
end

function muldiag!(A, v)
    for i in 1:size(A, 1)
        A[i, i] *= v[i] 
    end
end

function setdiag!(A, v)
    for i in 1:size(A, 1)
        A[i, i] = v[i] 
    end
end

function nat_to_meanvar(η₁, η₂)
    invΣ = cholesky(- 2 * η₂)
    μ = invΣ \ η₁
    return μ, inv(invΣ.L)
end

function meanvar_to_nat(μ, L)
    invΣ = XXt(inv(L))
    invΣ * μ, -0.5 * invΣ
end

nat_to_expec(η₁, η₂) = meanvar_to_expec(nat_to_meanvar(η₁, η₂)...)

expec_to_nat(μ, E) = mean_var_to_nat(expec_to_meanvar(μ, E)...)

function expec_to_meanvar(μ, E)
    μ, cholesky(E - XXt(μ)).L
end

function meanvar_to_expec(μ, L)
    μ, XXt(L) + XXt(μ)
end

function cov_to_lowrank_plus_diag(S, K)
    L = cov_to_lowrank(S, K)
    D = diag(S) / sqrt(2f0)
    return L, D
end

function cov_to_lowrank(S::AbstractMatrix, K)
    Q = svd(S)
    L = Q.U[:, 1:K] * Diagonal(sqrt.(Q.S[1:K]))
    return Matrix(L)
end

function cov_to_lowrank(S::Diagonal, K)
    s = partialsortperm(S.diag, 1:K; rev=true)
    return Matrix(S[:, s])
end

function cov_to_inv_lowrank_plus_diag(S, K)
    P = inv(S)
    return cov_to_lowrank_plus_diag(P, K)
end


## Making the ScaleTransform GPU compatible, same for LinearKernel
using Functors
using KernelFunctions
struct GPUScaleTransform{T<:Real,V<:AbstractVector{T}} <: KernelFunctions.Transform
    s::V
end

function GPUScaleTransform(s::T=1.0) where {T<:Real}
    return GPUScaleTransform{T,typeof([s])}([s])
end

@functor GPUScaleTransform

KernelFunctions.set!(t::GPUScaleTransform, ρ::Real) = t.s .= ρ

(t::GPUScaleTransform)(x) = first(t.s) * x

KernelFunctions._map(t::GPUScaleTransform, x::AbstractVector{<:Real}) = first(t.s) .* x
KernelFunctions._map(t::GPUScaleTransform, x::ColVecs) = ColVecs(first(t.s) .* x.X)
KernelFunctions._map(t::GPUScaleTransform, x::RowVecs) = RowVecs(first(t.s) .* x.X)

Base.isequal(t::GPUScaleTransform, t2::GPUScaleTransform) = isequal(first(t.s), first(t2.s))

Base.show(io::IO, t::GPUScaleTransform) = print(io, "Scale Transform (s = ", first(t.s), ")")


struct MyLinearKernel <: KernelFunctions.SimpleKernel end

KernelFunctions.kappa(::MyLinearKernel, xᵀy::Real) = xᵀy + 1

KernelFunctions.metric(::MyLinearKernel) = KernelFunctions.DotProduct()

