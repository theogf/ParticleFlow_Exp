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


function _grad_logposterior(bnn_arch::BNNArchitecture, D::Data, θ::Parameters)
    gradient(x -> _unnormalized_logposterior(bnn_arch, D, x), θ)
end


# bnn_arch = BNNArchitecture([4])
# θ = rand(bnn_arch.prior)
# nn = _build_nn(bnn_arch, θ)
# X = [-20:20;] ./ 10
# y = nn.(eachrow(X))
# y = vcat(y...)
# D = Data(X, y)
#
# logpdf(bnn_arch.prior, θ)
#
# gradient(x -> logpdf(bnn_arch.prior, x), θ)
#
# _grad_logposterior(bnn_arch, D, θ)


##

min_val, max_val = -2.0, 2.0

function params_to_line(bnn_arch::BNNArchitecture, θ::Parameters)
    nn = _build_nn(bnn_arch, θ)
    x = [range(min_val, max_val; length=100);]
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

##
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

n_points = 20
θ_t = rand(bnn_arch.prior, n_points)
scene, θ_t_node = set_plotting_scene_bnn(bnn_arch, X, y, θ, θ_t)
##
# Integration with pflowbase
#
# struct BNN <: AbstractModel
#     D::Data
#     bnn_arch::BNNArchitecture
# end
#
# function _f(x, p::BNN)
#     length(x) == p.bnn_arch.num_parameters || error("Invalid number of params")
#
# end

# function create_model(bnn_arch::BNNArchitecture, D::Data)
#     GeneralModel(
#         (θ, params) -> logpdf(params[1].prior, θ),  # log prior
#         (θ, params) -> _unnormalized_loglikelihood(params[1], params[2], θ), # log likelihood
#         [bnn_arch, D]  # params
#     )
# end
#
# model = create_model(bnn_arch, Data(X, y))
#
# η = 1.0
# opt0 = [Flux.ADAGrad(η), Flux.ADAGrad(0.0)]
# opt1 = [Flux.ADAGrad(η), Flux.ADAGrad(η)]
#
# record(scene, plotsdir("gifs", "bnn_toy.gif"), 1:400; framerate = 10) do i
#     @info i
#     if i < 50
#         move_particles(θ_t_node[], model, opt0, Xt=θ_t_node[], precond_b=true, precond_A=false)
#     else
#         move_particles(θ_t_node[], model, opt1, Xt=θ_t_node[], precond_b=true, precond_A=false)
#     end
#     #x_p[] = x_t
#     Makie.update!(scene)
#     # ∇f_p[] = ∇f1
# end
##


using Turing
using Distributions, DistributionsAD
using AdvancedVI; const AVI = AdvancedVI
using Makie, StatsMakie, Colors, MakieLayout
using KernelFunctions, Flux, KernelDensity

max_iters = 2
# advi = AVI.ADVI(n_points, max_iters)
# adq = AVI.transformed(TuringDiagMvNormal([mu_init],[sig_init]),AVI.Bijectors.Identity{1}())
#
# quadvi = AVI.ADQuadVI(n_points, max_iters)
# θ_init = [mu_init,sig_init]
# quadq = AVI.transformed(TuringDiagMvNormal([mu_init],[sig_init]),AVI.Bijectors.Identity{1}())
#
# steinvi = AVI.SteinVI(max_iters, transform(SqExponentialKernel(), 1.0))
# steinq =
#     AVI.SteinDistribution(rand(Normal(mu_init, sqrt(sig_init)), 1, nParticles))

gaussvi = AVI.PFlowVI(max_iters, false, true)
gaussq = SamplesMvNormal(copy(θ_t))
# gaussq = AVI.transformed(SamplesMvNormal(rand(Normal(mu_init, sqrt(sig_init)),1,nParticles)),AVI.Bijectors.Identity{1}())

using ReverseDiff
setadbackend(:reversediff)
logπ_base(x) = _unnormalized_logposterior(bnn_arch, D, θ)#log(1/3*pdf(d1,first(x)) + 2/3*pdf(d2,first(x)))

#α =  0.01
#optgauss = ADAGrad(α)

α = 0.001
optgauss = ADAM(α)

t = Node(0)

pdfgauss = lift(t) do _
    if gaussq.Σ[1,1] > 0
        gaussqn = Normal(gaussq.μ[1],sqrt(gaussq.Σ[1,1]))
        pdfgauss = pdf.(Ref(gaussqn),xrange)
        #push!(θ_t_node, gaussq.x[:])
        θ_t_node[] = gaussq.x
        #gaussvi.
    else
        @info "Zero sigma"
        @info gaussq
    end
end

record(scene, joinpath(plotsdir(),"gifs","bnn_toy.gif"),framerate=25) do io
    for i in 1:500
        #global adq = AVI.vi(logπ_base, advi, adq, θ_init, optimizer = optad)
        # the following doesn't seem to compile
        #global quadq = AVI.vi(logπ_base, quadvi, quadq, θ_init, optimizer = optquad)
        #AVI.vi(logπ_base, steinvi, steinq, optimizer = optstein)
        AVI.vi(logπ_base, gaussvi, gaussq, optimizer = optgauss)
        if i % 10 == 0
            t[] = i
        end
        recordframe!(io)
    end
end
