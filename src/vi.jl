using Distributions
using LinearAlgebra
using Random
using ForwardDiff: gradient, jacobian, derivative
using Flux: Optimise, destructure

export DSVI, FCS, NGD
export XXt, update!

abstract type VIScheme end

Distributions.MvNormal(d::VIScheme) = MvNormal(mean(d), cov(d))
nSamples(d::VIScheme) = d.nSamples

include("utils.jl")
include(joinpath("algs", "dsvi.jl"))
include(joinpath("algs", "fcs.jl"))
include(joinpath("algs", "gvar.jl"))
include(joinpath("algs", "ngd.jl"))
