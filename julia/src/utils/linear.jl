using MLDataUtils: rescale!, splitobs

function load_logistic_data(dataset)
    data = CSV.read(datadir("exp_raw", "gp", dataset*".csv"), DataFrame)
    X, y = Matrix(data) |> x->(x[:, 1:end-1], x[:, end])
    if unique(y) == [-1, 1]
        y[y.==-1] .= 0
    end
    rescale!(X; obsdim = 1)
    X = hcat(ones(size(X, 1)), X) # Add a constant term
    splitobs((X, y); at = 0.66, obsdim = 1)
end
