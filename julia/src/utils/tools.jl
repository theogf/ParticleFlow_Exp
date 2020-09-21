using Distances

# Return true for i = 1,2,3,..10,20,30,..,100,200 etc
iseverylog10(iter) = iter % (10^(floor(Int64, log10(iter + 1)))) == 0


# Return the mean and the variance over all particles
function mean_and_var(f::Function, q::AVI.AbstractSamplesMvNormal)
    vals = f.(eachcol(q.x))
    return mean(vals), var(vals)
end

function initial_lengthscale(X)
    if size(X,1) > 10000
        D = pairwise(SqEuclidean(),X[StatsBase.sample(1:size(X,1),10000,replace=false),:],dims=1)
    else
        D = pairwise(SqEuclidean(),X,dims=1)
    end
    return sqrt(mean([D[i,j] for i in 2:size(D,1) for j in 1:(i-1)]))
end
