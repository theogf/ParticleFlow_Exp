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
using IterTools

## Load the data
new_size = 10
train_x, train_y = CIFAR10.traindata()
train_x = reduce(hcat, map(x->vec(Float32.(Gray.(imresize(x, new_size, new_size)))), eachslice(CIFAR10.convert2image(train_x), dims=3)))
train_y = Flux.onehotbatch(train_y, 0:9)
test_x, test_y = CIFAR10.testdata()
test_x = reduce(hcat, map(x->vec(Float32.(Gray.(imresize(x, new_size, new_size)))), eachslice(CIFAR10.convert2image(test_x), dims=3)))
test_y = Flux.onehotbatch(test_y, 0:9)

## Create a NN
l1 = Dense(new_size ^ 2, 10, relu)
l2 = Dense(10, 10)
nn = Chain(l1, l2)
loss(x, y, nn=nn) = Flux.Losses.logitcrossentropy(nn(x), y)
ps = Flux.params(nn)
bs = 300
N = size(train_x, 2)
data = Flux.Data.DataLoader((train_x, train_y), batchsize=bs, shuffle=true)
accuracy(x, y, m) = mean(Flux.onecold(cpu(m(x)), 0:9) .== Flux.onecold(cpu(y), 0:9))



## Train the model
Flux.@epochs 100 train!(loss, ps, data, ADAM())
@info "Accuracy : $(accuracy(train_x, train_y, nn))"
@info "Loss : $(loss(train_x, train_y))"

## Destructure the model
θ, re = Flux.destructure(nn)
n_θ = length(θ)
D = n_θ
prior = MvNormal(zeros(D), 10)
function stochlogπ(x, y)
    return θ -> -Flux.Losses.logitcrossentropy(re(θ)(x), y) * N / bs + logpdf(prior, θ)
end
logπ(θ) = -Flux.Losses.logitcrossentropy(re(θ)(train_x), train_y) + logpdf(prior, θ)

## 
C₀ = Matrix{Float32}(I(D))
μ₀ = zeros(Float32, D)
μ₀ = θ
# Run alg
η = 0.1
S = 2
n_epochs = 10
T = n_epochs * length(data)

Stest = 10
NGmu = true # Preconditionner on the mean
algs = Dict()
# algs[:dsvi] = DSVI(copy(μ₀), cholesky(C₀).L, S)
# algs[:fcs] = FCS(copy(μ₀), Matrix(sqrt(0.5) * Diagonal(cholesky(C₀).L)), sqrt(0.5) * ones(D), S)
algs[:gpf] = GPF(rand(MvNormal(μ₀, C₀), S), NGmu)
algs[:gf] = GF(copy(μ₀), rand(Float32, D, S), S, NGmu)
# algs[:gf] = GF(copy(μ₀), Matrix{Float32}(cholesky(C₀).L), S, NGmu)
# algs[:iblr] = IBLR(copy(μ₀), inv(C₀), S)

# algs[:spm] = SPM(copy(μ₀), inv(cholesky(C₀).L), S)
# algs[:ngd] = NGD(copy(μ₀), cholesky(C₀).L)
(x_batch, y_batch) = first(data)

ELBOs = Dict()
train_accs = Dict()
test_accs = Dict()
train_nlls = Dict()
test_nlls = Dict()
# err_cov = Dict()
times = Dict()
opts = Dict()
function accuracy(x, y, d::VIScheme; usemean=false, Stest=10)
    mean(accuracy(x, y, re(θ)) for θ in eachcol(rand(d, Stest)))
end

function nll(x, y, d::VIScheme; usemean=false, Stest=10)
    mean(loss(x, y, re(θ)) for θ in eachcol(rand(d, Stest)))
end

for (name, alg) in algs
    ELBOs[name] = zeros(T+1)
    ELBOs[name][1] = ELBO(alg, logπ, nSamples = Stest)
    train_accs[name] = zeros(T+1)
    train_accs[name][1] = accuracy(train_x, train_y, alg)
    test_accs[name] = zeros(T+1)
    test_accs[name][1] = accuracy(test_x, test_y, alg)
    train_nlls[name] = zeros(T+1)
    train_nlls[name][1] = nll(train_x, train_y, alg)
    test_nlls[name] = zeros(T+1)
    test_nlls[name][1] = nll(test_x, test_y, alg)
    times[name] = 0
    opts[name] = RMSProp(η)
end


opts[:gpf] = Descent(η)
opts[:gpf] = MatRMSProp(η)
opts[:iblr] = Descent(η)
i = 1
@showprogress for (x, y) in IterTools.ncycle(data, n_epochs)
    f = stochlogπ(x, y)
    # stochlogπ(θ) = -Flux.Losses.logitcrossentropy(re(θ)(x), y) * N / bs + logpdf(prior, θ)
    for (name, alg) in algs
        t = @elapsed update!(alg, f, opts[name])
        times[name] += t
        ELBOs[name][i+1] = ELBO(alg, stochlogπ, nSamples = Stest)
        train_accs[name][i + 1] = accuracy(train_x, train_y, re(mean(alg)))
        test_accs[name][i + 1] = accuracy(test_x, test_y, re(mean(alg)))
        train_nlls[name][i + 1] = Flux.Losses.logitcrossentropy(re(mean(alg))(train_x), train_y)
        test_nlls[name][i + 1] = Flux.Losses.logitcrossentropy(re(mean(alg))(test_x), test_y)
    end
    global i += 1
end
for (name, alg) in algs
    @info "$name :\nELBO = $(ELBO(alg, logπ, nSamples = Stest))\nTime : $(times[name])"
end
## Profiling
name = :gpf
alg = algs[:gpf]
@profview ELBO(alg, stochlogπ, nSamples= Stest)
@time update!(alg, stochlogπ, opts[name])
update!(alg, stochlogπ, opts[name])
@profview update!(alg, stochlogπ, opts[name])

## Plotting difference
using Plots
pyplot()
p_L = plot(title = "ELBO")
for (name, alg) in algs
    cut = findfirst(x->x==0, ELBOs[name])
    cut = isnothing(cut) ? T : cut
    plot!(p_L, 1:cut, ELBOs[name][1:cut], lab = acs[name])
end
p_tr_acc = plot(title = "Train Acc")
for (name, alg) in algs
    cut = findfirst(x->x==0, train_accs[name])
    cut = isnothing(cut) ? T : cut
    plot!(p_tr_acc, 1:cut, train_accs[name][1:cut], lab = acs[name])
end
p_te_acc = plot(title = "Test Acc")
for (name, alg) in algs
    cut = findfirst(x->x==0, test_accs[name])
    cut = isnothing(cut) ? T : cut
    plot!(p_te_acc, 1:cut, test_accs[name][1:cut], lab = acs[name])
end
p_tr_nll = plot(title = "Train NLL")
for (name, alg) in algs
    cut = findfirst(x->x==0, train_nlls[name])
    cut = isnothing(cut) ? T : cut
    plot!(p_tr_nll, 1:cut, train_nlls[name][1:cut], lab = acs[name])
end
p_te_nll = plot(title = "Test NLL")
for (name, alg) in algs
    cut = findfirst(x->x==0, test_nlls[name])
    cut = isnothing(cut) ? T : cut
    plot!(p_te_nll, 1:cut, test_nlls[name][1:cut], lab = acs[name])
end
p_all = plot(p_L, p_tr_acc, p_te_acc, p_tr_nll, p_te_nll)
p_all |> display
opt_s = nameof(typeof(opts[:gf]))
opt_d = nameof(typeof(opts[:gpf]))
savefig(plotsdir("CIFAR10 - " * @savename(D, S, NGmu, opt_s, opt_d, η, new_size, bs, n_epochs) * ".png"))
