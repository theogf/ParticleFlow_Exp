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