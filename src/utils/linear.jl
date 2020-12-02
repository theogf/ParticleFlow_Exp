using MLDataUtils: rescale!, splitobs
using CSV
using StatsFuns: logistic
using StatsBase
using DataDeps, DataFrames, Tables

function treat_swarm(file_path)
    run(`unzip $file_path -d $(dirname(file_path))`)
    dir_path = dirname(file_path)
    data = CSV.read(joinpath(dir_path, "Swarm Behavior Data", "Flocking.csv"), DataFrame)
    data[end, 1] = "1407.1" # The data is corrupted and is missing one value
    data.x1 = parse.(Float64, data.x1)
    data = data[shuffle(axes(data, 1)), :] # Shuffle the data
    CSV.write(datadir("exp_raw", "linear", "swarm_flocking.csv"), data)
end

function treat_bioresponse(file_path)
    data = CSV.read(file_path, DataFrame)
    data = data[shuffle(axes(data, 1)), :] # Shuffle the data
    CSV.write(datadir("exp_raw", "linear", "bioresponse.csv"), data)
end

register(DataDep("swarm_flocking",
    """
    Swarm Flocking dataset :
    Coming from the UCI repository : https://archive.ics.uci.edu/ml/datasets/Swarm+Behaviour
    24016 Samples and 24000 Features with binary labels
    """,
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00524/Swarm%20Behavior%20Data.zip";
    post_fetch_method = treat_swarm)
)

register(DataDep("bioresponse",
    """
    BioResponse dataset :
    Coming from the Kaggle challenge : https://www.kaggle.com/c/bioresponse
    3751 Samples and 1776 Features with binary labels
    """,
    "https://www.openml.org/data/get_csv/1681097/phpSSK7iA";
    post_fetch_method = treat_bioresponse)
)

function load_logistic_data(dataset)
    data = CSV.read(datadir("exp_raw", "linear", dataset * ".csv"), DataFrame)
    X, y = Matrix(data) |> x->(x[:, 1:end-1], x[:, end])
    if unique(y) == [-1, 1]
        y[y.==-1] .= 0
    end
    rescale!(X; obsdim = 1)
    X = hcat(ones(size(X, 1)), X) # Add a constant term
    splitobs((X, y); at = 0.8, obsdim = 1)
end
