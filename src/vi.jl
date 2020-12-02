module VariationalInference

using Distributions
using LinearAlgebra
using ForwardDiff: gradient, hessian
using Flux.Optimise

abstract type VIScheme end

MvNormal(d::VIScheme) = MvNormal(mean(d), cov(d))
nSamples(d::VIScheme) = d.nSamples

include(joinpath("algs", "dsvi.jl"))
include(joinpath("algs", "gvar.jl"))

end