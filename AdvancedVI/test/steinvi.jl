using AdvancedVI; const AVI = AdvancedVI
using Turing
using Flux
using KernelFunctions, Distances
using ForwardDiff
using LinearAlgebra


@model model(x) = begin
    s ~ InverseGamma(2, 3)
    m ~ Normal(0.0, sqrt(s))
    for i = 1:length(x)
        x[i] ~ Normal(m, sqrt(s))
    end
end

x = randn(200)

max_iter = 20
k = transform(SqExponentialKernel(),1.0)
m = model(x)
steinvi = AVI.SteinVI(max_iter, k)
q = AdvancedVI.vi(m, steinvi, 10, optimizer = ADAGrad(0.1))
mean(q)
cov(q)
var(q)
# global q = AdvancedVI.vi(m, steinvi, q, optimizer = ADAGrad(0.1))
