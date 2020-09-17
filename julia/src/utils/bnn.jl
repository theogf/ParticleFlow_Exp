@with_kw mutable struct Args
    η = 3e-4             # learning rate
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize = 128      # batch size
    epochs = 20          # number of epochs
    seed = 0             # set seed > 0 for reproducibility
    cuda = true          # if true use cuda (if available)
    infotime = 1      # report every `infotime` epochs
    checktime = 5        # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    tblogger = false      # log training with tensorboard
    savepath = nothing    # results path. If nothing, construct a default path from Args. If existing, may overwrite
    datapath = joinpath(homedir(), "Datasets", "MNIST") # data path: change to your data directory
end
function get_data(args)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32, dir = args.datapath)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32, dir = args.datapath)

    xtrain = reshape(xtrain, 28, 28, 1, :)
    xtest = reshape(xtest, 28, 28, 1, :)

    ytrain, ytest = Flux.onehotbatch(ytrain, 0:9), Flux.onehotbatch(ytest, 0:9)

    train_loader =
        Flux.Data.DataLoader((xtrain, ytrain); batchsize = args.batchsize, shuffle = true)
    test_loader = Flux.Data.DataLoader((xtest, ytest); batchsize = args.batchsize)

    return train_loader, test_loader
end
