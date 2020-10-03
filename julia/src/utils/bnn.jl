using Flux
using Random
using MLDatasets
using ProgressMeter
function get_data(dataset, batchsize)
    if dataset == "MNIST"
        return get_MNIST_data(batchsize)
    else
        error("Unknown dataset $dataset")
    end
end

function get_MNIST_data(batchsize)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32, dir = datadir("data_raw", "MNIST"))
    xtest, ytest = MLDatasets.MNIST.testdata(Float32, dir = datadir("data_raw", "MNIST"))

    xtrain = reshape(xtrain, 28, 28, 1, :)
    xtest = reshape(xtest, 28, 28, 1, :)

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
