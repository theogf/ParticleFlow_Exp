using AugmentedGaussianProcesses
using CSV
using Distances
using DataFrames: DataFrame
using MLDataUtils: rescale!, splitobs

function load_gp_data(dataset)
    data = CSV.read(datadir("exp_raw", "gp", dataset*".csv"), DataFrame)
    X, y = Matrix(data) |> x->(x[:, 1:end-1], x[:, end])
    if unique(y) == [-1, 1]
        y[y.==-1] .= 0
    end
    rescale!(X; obsdim = 1)
    splitobs((X, y); at = 0.66, obsdim = 1)
end


function initial_lengthscale(X)
    if size(X,1) > 10000
        D = pairwise(SqEuclidean(),X[StatsBase.sample(1:size(X,1),10000,replace=false),:],dims=1)
    else
        D = pairwise(SqEuclidean(),X,dims=1)
    end
    return sqrt(mean([D[i,j] for i in 2:size(D,1) for j in 1:(i-1)]))
end
