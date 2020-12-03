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
  Distributions._rand!(rng, MvNormal(d), x)
end

function Random.rand(d::VIScheme, n::Int)
  Distributions._rand!(Random.GLOBAL_RNG, d, zeros(dim(d), n))
end

nSamples(d::VIScheme) = d.nSamples

include("utils.jl")
include(joinpath("algs", "dsvi.jl"))
include(joinpath("algs", "fcs.jl"))
include(joinpath("algs", "gf.jl"))
include(joinpath("algs", "gpf.jl"))
include(joinpath("algs", "gvar.jl"))
include(joinpath("algs", "ngd.jl"))
include(joinpath("algs", "spm.jl"))
