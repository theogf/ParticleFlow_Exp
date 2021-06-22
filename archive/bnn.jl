using DrWatson
@quickactivate
include(srcdir("pflowbase.jl"))
#include(srcdir("makie_plotting.jl"))

using Flux
using Makie
using Distributions, LinearAlgebra
import Base.@kwdef

@kwdef struct BNNArchitecture
    input_size::Int
    output_size::Int
    hidden_layer_sizes::Vector{Int}
    num_parameters::Int
    prior::ContinuousMultivariateDistribution
    noise_sigma::Float16
end

# convenience constructor for simple 1 dim regression
function BNNArchitecture(hidden_layer_sizes::Vector{Int})
    input_size = 1  # for now
    output_size = 1

    # compute num of parameters
    prev_layer_size = input_size
    num_parameters = 0
    for layer_size in vcat(hidden_layer_sizes, [output_size])
        num_parameters += (prev_layer_size+1)*layer_size
        prev_layer_size = layer_size
    end

    BNNArchitecture(
        input_size=1,
        output_size=1,
        hidden_layer_sizes=hidden_layer_sizes,
        num_parameters=num_parameters,
        prior=MvNormal(zeros(num_parameters), I),
        noise_sigma=0.1
    )
end

struct Data
    X::Vector
    y::Vector
end

const Parameters = Vector


# Un flatten parameters putting them as weight in a neural net
function _build_nn(bnn_arch::BNNArchitecture, θ::Parameters)
    @assert bnn_arch.num_parameters == length(θ) (bnn_arch.num_parameters, length(θ))
    i = 1
    layers = []
    all_layer_sizes = vcat([bnn_arch.input_size], bnn_arch.hidden_layer_sizes, [bnn_arch.output_size])
    prev_layer_size = bnn_arch.input_size
    for layer_size in bnn_arch.hidden_layer_sizes
        num_weights = layer_size * prev_layer_size
        W = reshape(θ[i:i+num_weights-1], layer_size, prev_layer_size)
        i += num_weights
        b = θ[i:i+layer_size-1]
        layers = vcat(layers, [Dense(W, b, tanh)])
        prev_layer_size = layer_size
        i += layer_size
    end
    layer_size = bnn_arch.output_size
    num_weights = layer_size * prev_layer_size
    W = reshape(θ[i:i+num_weights-1], layer_size, prev_layer_size)
    i += num_weights
    b = θ[i:i+layer_size-1]
    @assert i+layer_size-1 == length(θ) (i, layer_size, length(θ))
    layers = vcat(layers, [Dense(W, b, identity)])
    Chain(layers...)
end

#_numlayers(bnnarch::BNNArchitecture) = length(bnnarch)

function _unnormalized_loglikelihood(bnn_arch::BNNArchitecture, D::Data, θ::Parameters)
    nn = _build_nn(bnn_arch, θ)
    mse = sum((nn([x])[1] - y)^2 for (x, y) in zip(D.X, D.y))
    return -mse * bnn_arch.noise_sigma
end

function _unnormalized_logposterior(bnn_arch::BNNArchitecture, D::Data, θ::Parameters)
    _unnormalized_loglikelihood(bnn_arch, D, θ) + logpdf(bnn_arch.prior, θ)
end
#
#
# function _grad_logposterior(bnn_arch::BNNArchitecture, D::Data, θ::Parameters)
#     gradient(x -> _unnormalized_logposterior(bnn_arch, D, x), θ)
# end


##

min_val, max_val = -2.0, 2.0
xrange = [range(min_val, max_val; length=100);]

function params_to_line(bnn_arch::BNNArchitecture, θ::Parameters)
    nn = _build_nn(bnn_arch, θ)
    x = xrange
    y = [nn([x])[1] for x in x]
    x, y
end

function set_plotting_scene_bnn(bnn_arch::BNNArchitecture, X, y, θ, θ_t)
    θ_t_node = Node(θ_t)
    scene = Makie.scatter(X, y)

    # plot ground truth
    x, y = params_to_line(bnn_arch, θ)
    Makie.lines!(scene, x, y, linewidth=2.0, color = :green)

    for i in 1:size(θ_t, 2)
        xy_node = @lift params_to_line(bnn_arch, $θ_t_node[:, i])
        x_node = @lift $xy_node[1]
        y_node = @lift $xy_node[2]

        Makie.lines!(scene, x_node, y_node, linewidth=0.2)
    end
    scene, θ_t_node
end

## Build BNN model

bnn_arch = BNNArchitecture([4])
# groud truth
θ = rand(bnn_arch.prior)
# training points
num_inputs = 10
nn = _build_nn(bnn_arch, θ)
noise = rand(MvNormal(zeros(num_inputs), I*bnn_arch.noise_sigma))
X = rand(Float16, num_inputs) .* (max_val-min_val) .+ min_val
y = [nn([x])[1] for x in X]
y += noise

D = Data(X, y)

n_points = 20
θ_t = rand(bnn_arch.prior, n_points)
scene, θ_t_node = set_plotting_scene_bnn(bnn_arch, X, y, θ, θ_t)


##
using Turing
using Distributions, DistributionsAD
using AdvancedVI; const AVI = AdvancedVI
using Makie, StatsMakie, Colors, MakieLayout
using KernelFunctions, Flux, KernelDensity

max_iters = 2

gaussvi = AVI.PFlowVI(max_iters, false, true)
gaussq = SamplesMvNormal(copy(θ_t))

using ReverseDiff
setadbackend(:reversediff)
logπ_base(x) = _unnormalized_logposterior(bnn_arch, D, x)#log(1/3*pdf(d1,first(x)) + 2/3*pdf(d2,first(x)))

α =  0.1
optgauss = ADAGrad(α)

# α = 0.001
# optgauss = ADAM(α)

t = Node(0)

# should probably just be an event somehow
# updates θ_t_node, everytime t changes
lift(t) do _
    if gaussq.Σ[1,1] > 0
        gaussqn = Normal(gaussq.μ[1], sqrt(gaussq.Σ[1,1]))
        pdfgauss = pdf.(Ref(gaussqn), xrange)
        #push!(θ_t_node, gaussq.x[:])
        θ_t_node[] = gaussq.x
        #gaussvi.
    else
        @info "Zero sigma"
        @info gaussq
    end
end

record(scene, joinpath(plotsdir(),"gifs","bnn_toy.gif"),framerate=25) do io
    for i in 1:100
        #global adq = AVI.vi(logπ_base, advi, adq, θ_init, optimizer = optad)
        # the following doesn't seem to compile
        #global quadq = AVI.vi(logπ_base, quadvi, quadq, θ_init, optimizer = optquad)
        #AVI.vi(logπ_base, steinvi, steinq, optimizer = optstein)
        AVI.vi(logπ_base, gaussvi, gaussq, optimizer = optgauss)
        if i % 5 == 0
            t[] = i
        end
        recordframe!(io)
    end
end
