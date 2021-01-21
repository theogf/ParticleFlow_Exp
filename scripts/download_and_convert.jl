using DrWatson
@quickactivate
using DataFrames
using CSV
using BSON
using MLDataUtils
using HTTP
using Random
using ProgressMeter
using LinearAlgebra

function categorical_to_binary(x)
    X = convertlabel(LabelEnc.OneOfK{Float64}, x, obsdim=1)
    return size(X,2) == 2 ? X[:, 1] : X[:, 1:end-1]
end
exp_dir = datadir("exp_raw")

## Processing datasets for logistic experiments

isdir(joinpath(exp_dir, "logistic")) ? nothing : mkpath(joinpath(exp_dir, "logistic"))

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

## Creating initial / target Gaussian
gauss_dir = datadir("exp_raw", "gaussian")
mkpath(gauss_dir)

@showprogress for cond in [1, 5, 10, 50, 100]
    for n_dim in [10, 20, 50, 100, 500, 1000]
        @info "Cond = $cond, Dim = $n_dim"
        file_name = @savename(n_dim, cond) * ".bson"
        if isfile(joinpath(gauss_dir, file_name)) # If file exists already we skip the process
            continue
        end
        μ_target = randn(n_dim)
        Σ_target = if cond > 1
            Q, _ = qr(rand(n_dim, n_dim)) # Create random unitary matrix
            Q = Matrix(Q)
            Λ = Diagonal(10.0 .^ range(-1, -1 + log10(cond), length = n_dim))
            Symmetric(Q * Λ * Q')
        else
            Matrix(I(n_dim))
        end
        α = eps(Float64)
        while !isposdef(Σ_target)
            Σ_target = Σ_target + α * I
            α *= 2
        end
        @info "Target done"
        μs_init = [randn(n_dim) for _ in 1:10]
        Σs_init = [ begin 
            Q, _ = qr(rand(n_dim, n_dim)) # Create random unitary matrix
            Q = Matrix(Q)
            Λ = Diagonal(exp.(randn(n_dim)))
            Symmetric(Q * Λ * Q')
        end for _ in 1:10]
        @info "Init done"
        DrWatson.save(joinpath(gauss_dir, file_name), @dict(μ_target, Σ_target, μs_init, Σs_init))
    end
end