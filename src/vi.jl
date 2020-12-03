using Distributions
using LinearAlgebra
using Random
using ForwardDiff: gradient, jacobian, derivative
using Flux: Optimise, destructure

export DSVI, FCS, NGD
export XXt, update!

abstract type VIScheme end

Distributions.MvNormal(d::VIScheme) = MvNormal(mean(d), cov(d))
function Distributions._rand!(
  rng::AbstractRNG,
  d::VIScheme,
  x::AbstractVecOrMat,
)
  _rand!(rng, MvNormal(d), x)
end
nSamples(d::VIScheme) = d.nSamples

include("utils.jl")
include(joinpath("algs", "dsvi.jl"))
include(joinpath("algs", "fcs.jl"))
include(joinpath("algs", "gvar.jl"))
include(joinpath("algs", "ngd.jl"))
include(joinpath("algs", "gpf.jl"))
include(joinpath("algs", "gf.jl"))
