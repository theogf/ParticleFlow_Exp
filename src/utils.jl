invquad(A::AbstractMatrix, x::AbstractVecOrMat) = dot(x, A \ x)
XXt(X::AbstractVecOrMat) = X * X'
gradcol(f, X) = gradient(x->sum(f.(eachcol(x))), X)

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

