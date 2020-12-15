using DrWatson
@quickactivate
include(srcdir("vi.jl"))
include(srcdir("utils", "dicts.jl"))
include(srcdir("utils", "optimisers.jl"))
include(srcdir("utils", "tools.jl"))
using Distributions, LinearAlgebra, Random
using ProgressMeter
using Flux.Optimise
using Flux
using Images, ImageTransformations
using MLDatasets

## Load the data
new_size = 5
train_x, train_y = CIFAR10.traindata()
train_x = reduce(hcat, map(x->vec(Float32.(Gray.(imresize(x, new_size, new_size)))), eachslice(CIFAR10.convert2image(train_x), dims=3)))
train_y = Flux.onehotbatch(train_y ,0:9)
test_x, test_y = CIFAR10.testdata()
test_x = reduce(hcat, map(x->vec(Float32.(Gray.(imresize(x, new_size, new_size)))), eachslice(CIFAR10.convert2image(test_x), dims=3)))
test_y = Flux.onehotbatch(test_y ,0:9)

## Create a NN
l1 = Dense(new_size ^ 2, 5, relu)
l2 = Dense(5, 10)
nn = Chain(l1, l2)
loss(x, y) = Flux.Losses.logitcrossentropy(nn(x), y)
ps = Flux.params(nn)
bs = 5
N = size(train_x, 2)
data = Flux.Data.DataLoader((train_x, train_y), batchsize=bs, shuffle=true)
accuracy(x, y, m) = mean(Flux.onecold(cpu(m(x)), 0:9) .== Flux.onecold(cpu(y), 0:9))

train!(loss, ps, data, ADAM())
@info accuracy(train_x, train_y, nn)
@info loss(train_x, train_y)
θ, re = Flux.destructure(nn)
n_θ = length(θ)
D = n_θ
prior = MvNormal(zeros(D), 10)
stochlogπ(θ) = -Flux.Losses.logitcrossentropy(re(θ)(x), y) * N / bs + logpdf(prior, θ)
logπ(θ) = -Flux.Losses.logitcrossentropy(re(θ)(train_x), train_y) + logpdf(prior, θ)

## 
C₀ = Matrix(I(D))
μ₀ = zeros(D)
## Run alg
η = 0.01
S = 100
Stest = 10
NGmu = true # Preconditionner on the mean
algs = Dict()
algs[:dsvi] = DSVI(copy(μ₀), cholesky(C₀).L, S)
algs[:fcs] = FCS(copy(μ₀), Matrix(sqrt(0.5) * Diagonal(cholesky(C₀).L)), sqrt(0.5) * ones(D), S)
algs[:gpf] = GPF(rand(MvNormal(μ₀, C₀), S), NGmu)
algs[:gf] = GF(copy(μ₀), Matrix(cholesky(C₀).L), S, NGmu)
algs[:iblr] = IBLR(copy(μ₀), inv(C₀), S)

# algs[:spm] = SPM(copy(μ₀), inv(cholesky(C₀).L), S)
# algs[:ngd] = NGD(copy(μ₀), cholesky(C₀).L)
(x,y) = first(data)

ELBOs = Dict()
train_accs = Dict()
test_accs = Dict()
train_nlls = Dict()
test_nlls = Dict()
# err_cov = Dict()
times = Dict()
opts = Dict()
for (name, alg) in algs
    ELBOs[name] = zeros(T+1)
    ELBOs[name][1] = ELBO(alg, logπ, nSamples = Stest)
    train_accs[name] = zeros(T+1)
    train_accs[name][1] = accuracy(train_x, train_y, re(mean(alg)))
    test_accs[name] = zeros(T+1)
    test_accs[name][1] = accuracy(test_x, test_y, re(mean(alg)))
    train_nlls[name] = zeros(T+1)
    train_nlls[name][1] = Flux.Losses.logitcrossentropy(re(mean(alg))(train_x), train_y)
    test_nlls[name] = zeros(T+1)
    test_nlls[name][1] = Flux.Losses.logitcrossentropy(re(mean(alg))(test_x), test_y)
    times[name] = 0
    opts[name] = RMSProp(η)
end


opts[:gpf] = Descent(η)
opts[:gpf] = MatRMSProp(η)
opts[:iblr] = Descent(η)
@showprogress for (i, (x, y)) in enumerate(data)
    for (name, alg) in algs
        t = @elapsed update!(alg, stochlogπ, opts[name])
        times[name] += t
        ELBOs[name][i+1] = ELBO(alg, logπ, nSamples = Stest)
        train_accs[name][i + 1] = accuracy(train_x, train_y, re(mean(alg)))
        test_accs[name][i + 1] = accuracy(test_x, test_y, re(mean(alg)))
        train_nlls[name][i + 1] = Flux.Losses.logitcrossentropy(re(mean(alg))(train_x), train_y)
        test_nlls[name] = zeros(T+1)
        test_nlls[name][i + 1] = Flux.Losses.logitcrossentropy(re(mean(alg))(test_x), test_y)
    end
end
for (name, alg) in algs
    @info "$name :\nELBO = $(ELBO(alg, logπ, nSamples = Stest))\nTime : $(times[name])"
end

# Plotting difference
using Plots
p_L = plot(title = "ELBO")
for (name, alg) in algs
    cut = findfirst(x->x==0, ELBOs[name])
    cut = isnothing(cut) ? T : cut
    plot!(p_L, 1:cut, ELBOs[name][1:cut], lab = acs[name])
end

p_L |> display
opt_s = nameof(typeof(opts[:gf]))
opt_d = nameof(typeof(opts[:gpf]))
savefig(plotsdir("Banana - " * @savename(S, NGmu, opt_s, opt_d, η) * ".png"))
## Plot the final status
lim = 20
xrange = range(-lim, lim, length = 200)
yrange = range(-lim, lim, length = 200)
ptruth = contour(xrange, yrange, banana, title = "truth", colorbar=false)
ps = [ptruth]
for (name, alg) in algs
    p = contour(xrange, yrange, (x,y)->pdf(MvNormal(alg), [x, y]), title = acs[name], colorbar=false)
    if alg isa GPF
        scatter!(p, eachrow(alg.X)..., lab="", msw=0.0, alpha = 0.6)
    end
    push!(ps, p)
end
plot(ps...) |> display

## Showing evolution 

# q = GPF(rand(MvNormal(μ₀, C₀), S), NGmu)
# a = Animation()
# opt = Momentum(0.1)
# opt = Descent(1.0)
# opt = MatADAGrad(1.0)
# @showprogress for i in 1:200
#     if i % 10 == 0
#         p = contour(xrange, yrange, logbanana, title = "i=$i", colorbar=false)
#         contour!(p, xrange, yrange, (x,y)->logpdf(MvNormal(q), [x, y]), colorbar=false)
#         scatter!(p, eachrow(q.X)..., lab="", msw=0.0, alpha = 0.9)
#         frame(a)
#     end
#     update!(q, logπ, opt)
# end
# gif(a, plotsdir("Banana - Momentum.gif"), fps = 10)