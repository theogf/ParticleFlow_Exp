using DrWatson
@quickactivate
using DataFrames
using CSV
using MLDataUtils
using HTTP

function categorical_to_binary(x)
    X = convertlabel(LabelEnc.OneOfK{Float64}, x, obsdim=1)
    return size(X,2) == 2 ? X[:, 1] : X[:, 1:end-1]
end
exp_dir = datadir("exp_raw")

## Processing datasets for logistic experiments

isdir(joinpath(exp_dir, "logistic")) ? mkpath(joinpath(exp_dir, "logistic")) : nothing

## Processing Mushroom dataset

local_path = datadir("exp_raw", "logistic", "mushroom.data")

HTTP.download("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
                local_path)

data = CSV.read(local_path, DataFrame)
data = hcat(data[2:end], data[1]) # Move label at the end
data = hcat(categorical_to_binary.(eachcol(data))...)
data = Tables.table(shuffleobs(data, obsdim=1))
CSV.write(joinpath(exp_dir, "logistic", "mushroom.csv"), data)

## Processing Ionopshere data

local_path = datadir("exp_raw", "logistic", "ionosphere.data")

HTTP.download("https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
                local_path)

data = CSV.read(local_path, DataFrame, header=false)
labels = categorical_to_binary(data[end])
data = hcat(data[1:end-1], labels)
data = Tables.table(shuffleobs(Matrix(data), obsdim=1))
CSV.write(joinpath(exp_dir, "logistic", "ionosphere.csv"), data)


## Processing KRKP

local_path = datadir("exp_raw", "logistic", "krkp.data")

HTTP.download("https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data",
                local_path)

data = CSV.read(local_path, DataFrame)
data = hcat(categorical_to_binary.(eachcol(data))...)
data = Tables.table(shuffleobs(data, obsdim=1))
CSV.write(joinpath(exp_dir, "logistic", "krkp.csv"), data)

## Processing Spam

local_path = datadir("exp_raw", "logistic", "spam.data")

HTTP.download("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
                local_path)

data = CSV.read(local_path, DataFrame, header=false)
data = Tables.table(shuffleobs(Matrix(data), obsdim=1))
CSV.write(joinpath(exp_dir, "logistic", "spam.csv"), data)
