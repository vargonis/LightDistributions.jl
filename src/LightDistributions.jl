module LightDistributions

using Random
using StatsFuns: log2π, poisinvcdf, poislogpdf, gammalogpdf
using SpecialFunctions: loggamma
using StaticArrays
using CUDAnative
using CuArrays

import Base: rand
export Distribution
export Categorical, Poisson, Uniform, Normal, Exponential, Gamma
export Dirichlet
export params, logpdf


abstract type Distribution{T,P} end

Base.eltype(::Distribution{T}) where T = T
Base.eltype(::Type{<:Distribution{T}}) where T = T
params(d::Distribution) = d.params

include("scalars.jl")
include("arrays.jl")

CuArrays.@cufunc logpdf(args...) = _logpdf(args...)


end # module
