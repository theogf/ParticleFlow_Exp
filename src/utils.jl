invquad(A::AbstractMatrix, x::AbstractVecOrMat) = dot(x, A \ x)
XXt(X::AbstractVecOrMat) = X * X'
# gradcol(alg::VIScheme, f::Function, X::AbstractMatrix) = gradcol(ad(alg), f, X)
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
    D = diag(S) / sqrt(2)
    return L, D
end

function cov_to_lowrank(S, K)
    Q = svd(S)
    L = Q.U[:, 1:K] * Diagonal(sqrt.(Q.S[1:K]))
    return L
end

function cov_to_inv_lowrank_plus_diag(S, K)
    P = inv(S)
    return cov_to_lowrank_plus_diag(P, K)
end