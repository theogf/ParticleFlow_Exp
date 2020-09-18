function get_data(dataset, batchsize)
    if dataset == "MNIST"
        return get_MNIST_data(args)
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
