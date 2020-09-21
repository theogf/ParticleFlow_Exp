using AugmentedGaussianProcesses
using CSV, DataFrames
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
