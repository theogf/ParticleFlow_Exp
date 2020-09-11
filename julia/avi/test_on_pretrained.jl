cd(@__DIR__)
using Turing, Flux, Plots, Random;
# using ONNX
using AdvancedVI, Bijectors
using BSON
const AVI = AdvancedVI
using ReverseDiff
using ValueHistories
AVI.setadbackend(:reversediff)
model = "squeezenet1.1"
model = "mobilenetv2-1.0"
model = "lenet_mnist"
model_dir = joinpath("/home","theo","experiments","ParticleFlow","julia","pretrained_models",model)
# ONNX.load_model(joinpath(model_dir, model * ".onnx"))
# weights = ONNX.load_weights(joinpath(model_dir, "weights.bson"))
# model = include(joinpath(model_dir, "model.jl"))
m = BSON.load(joinpath(model_dir, "model.bson"))[:model]
convm = m[1:6]
densem = m[7:end]
dense_θ, dense_re = Flux.destructure(densem)
n_θ = length(dense_θ)


function nn_forward(xs, nn_params::AbstractVector)
    densem = dense_re(nn_params)
    nn = Chain(convm, densem)
    return nn(xs)
end;


## Create model

# Create a regularization term and a Gaussain prior variance term.
alpha = 0.09
sig = sqrt(1.0 / alpha)
args = Args()

train_loader, test_loader = get_data(args)
y = first(train_loader)

for (x, y) in train_loader
    x, y = x |> device, y |> device
    gs = Flux.gradient(ps) do
        ŷ = model(x)
        loss(ŷ, y)
    end
    Flux.Optimise.update!(opt, ps, gs)
    ProgressMeter.next!(p)   # comment out for no progress bar
end
# Specify the probabalistic model.
@model bayes_nn(xs, ys) = begin
    # Create the weight and bias vector.
    θ ~ MvNormal(zeros(n_θ), sig .* ones(n_θ))

    # Calculate predictions for the inputs given the weights
    # and biases in theta.
    preds = nn_forward(xs, θ)
    ys = onecold(ys)
    # Observe each prediction.
    for i = 1:length(ys)
        ys[i] ~ Categorical(softmax(preds[i]))
    end
end;
