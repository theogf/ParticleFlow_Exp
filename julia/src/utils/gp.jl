using CSV
using Distances
using DataFrames: DataFrame
using MLDataUtils: rescale!, shuffleobs, splitobs
using DataDeps, Tables

register(DataDep("ionosphere",
    """
    Ionosphere dataset :
    Coming from the UCI repository : https://archive.ics.uci.edu/ml/datasets/Ionosphere
    351 Samples and 34 Features with binary labels
    """,
    "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data";
    post_fetch_method = treat_ionosphere)
)
function treat_ionosphere(datadeps_path)
    data = Matrix(CSV.read(datadeps_path, DataFrame; header=false))
    y = data[:, end]
    y[y.==Ref("b")] .= 0
    y[y.==Ref("g")] .= 1
    data = Float64.([data[:, 1:end-1] y])
    target_dir = datadir("exp_raw", "gp")
    CSV.write(joinpath(target_dir, "ionosphere.csv"), Tables.table(data))
    @info "Wrote file in $(target_dir)"
end



function load_gp_data(dataset)
    data = Matrix(CSV.read(datadir("exp_raw", "gp", dataset*".csv"), DataFrame))
    X, y = data[:, 1:end-1], data[:, end]
    if unique(y) == [-1, 1]
        y[y.==-1] .= 0
    end
    rescale!(X; obsdim = 1)
    # X, y = shuffleobs((X, y), obsdim = 1, rng = MersenneTwister(42))
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
