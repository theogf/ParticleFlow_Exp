using Flux
using DataFrames
using BSON
using Random
using MLDatasets
using ProgressMeter

function get_data(dataset, batchsize)
    if dataset == "MNIST"
        return get_MNIST_data(batchsize)
    elseif dataset == "CIFAR10"
        return get_CIFAR10_data(batchsize)
    else
        error("Unknown dataset $dataset")
    end
end

function get_MNIST_data(batchsize)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32, dir = datadir("data_raw", "MNIST"))
    xtest, ytest = MLDatasets.MNIST.testdata(Float32, dir = datadir("data_raw", "MNIST"))

    xtrain = reshape(xtrain, 28, 28, :)
    xtest = reshape(xtest, 28, 28, :)

    ytrain, ytest = Flux.onehotbatch(ytrain, 0:9), Flux.onehotbatch(ytest, 0:9)

    train_loader =
        Flux.Data.DataLoader((xtrain, ytrain); batchsize = batchsize, shuffle = true)
    test_loader = Flux.Data.DataLoader((xtest, ytest); batchsize = batchsize)

    return train_loader, test_loader
end

function get_CIFAR10_data(batchsize)
    xtrain, ytrain = MLDatasets.CIFAR10.traindata(Float32, dir = datadir("data_raw", "CIFAR"))
    xtest, ytest = MLDatasets.CIFAR10.testdata(Float32, dir = datadir("data_raw", "CIFAR"))

    xtrain = reshape(xtrain, 32, 32, 3, :)
    xtest = reshape(xtest, 32, 32, 3, :)

    ytrain, ytest = Flux.onehotbatch(ytrain, 0:9), Flux.onehotbatch(ytest, 0:9)

    train_loader =
        Flux.Data.DataLoader((xtrain, ytrain); batchsize = batchsize, shuffle = true)
    test_loader = Flux.Data.DataLoader((xtest, ytest); batchsize = batchsize)

    return train_loader, test_loader
end

function LeNet5(; imgsize=(28,28,1), nclasses=10)
    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)

    return Chain(
            x -> reshape(x, imgsize..., :),
            Conv((5, 5), imgsize[end]=>6, relu),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu),
            MaxPool((2, 2)),
            x -> reshape(x, :, size(x, 4)),
            Dense(prod(out_conv_size), 120, relu),
            Dense(120, 84, relu),
            Dense(84, nclasses)
          )
end

function MNIST_BNN(; imgsize=(28,28,1), nclasses= 10)
        return Chain(
                x -> reshape(x, prod(imgsize), :),
                Dense(prod(imgsize), 500, relu),
                Dense(500, 400, relu),
                Dense(400, nclasses)
              )
end

function std_BNN(; imgsize=(32, 32, 3), n_units=100, n_classes=10)
        return Chain(
                x -> reshape(x, prod(imgsize), :),
                Dense(prod(imgsize), n_units, relu),
                Dense(n_units, n_units, relu),
                Dense(n_units, n_classes)
              )
end



function vgg16()
    return Chain(
            Conv((3, 3), 3 => 64, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(64),
            Conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(64),
            MaxPool((2,2)),
            Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(128),
            Conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(128),
            MaxPool((2,2)),
            Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            MaxPool((2,2)),
            Conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            MaxPool((2,2)),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            MaxPool((2,2)),
            Flux.flatten,
            Dense(512, 4096, relu),
            Dropout(0.5),
            Dense(4096, 4096, relu),
            Dropout(0.5),
            Dense(4096, 10)) |> gpu
end

function eval_loss_accuracy(loader, model, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        l += loss(ŷ, y) * size(x)[end]
        acc += sum(Flux.onecold(ŷ |> cpu) .== Flux.onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (loss = l/ntot |> round4, acc = acc/ntot*100 |> round4)
end

loss(ŷ, y) = Flux.Losses.logitcrossentropy(ŷ, y)
round4(x) = round(x, digits=4)
logprior(θ::AbstractArray{<:Real}) = sum(abs2, θ)
logprior(θ, α) = - sum(logprior, θ) / α
function train_and_save(model_name, dataset; device = gpu, n_epoch = 100, α = 0.1)
    m = @eval $(Symbol(model_name))()
    m = m |> device
    ps = Flux.params(m)
    train_loader, test_loader = get_data(dataset, 128)
    nobs = train_loader.nobs
    opt = ADAM(0.001f0)
    for epoch in 1:n_epoch
        p = ProgressMeter.Progress(length(train_loader))
        for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                ŷ = m(x)
                nobs * loss(ŷ, y)  - logprior(ps, α)
            end
            Flux.Optimise.update!(opt, ps, gs)
            ProgressMeter.next!(p)
        end
        train = eval_loss_accuracy(train_loader, m, device)
        test = eval_loss_accuracy(test_loader, m, device)
        @info "Epoch $epoch/$(n_epoch), Train acc : $train, Test acc : $test"
    end
    path = projectdir("bnn_models", model_name)
    @info "Saving model in $path"
    ispath(path) ? nothing : mkpath(path)
    filepath = joinpath(path, savename("model", @dict α n_epoch), ".bson")
    BSON.@save filepath m
end


# train_and_save("CIFAR_BNN", "CIFAR10", n_epoch = 100)

"""
    simplebnn(nhidden, ninput, noutput, activation=tanh)

Create a simple Bayesian neural network with 2 hidden layers with `nhidden` units.
"""
function simplebnn(nhidden, ninput, noutput, activation=tanh)
    return Chain(
            Dense(ninput, nhidden, activation),
            Dense(nhidden, nhidden, activation),
            Dense(nhidden, noutput, identity)
    )
end


