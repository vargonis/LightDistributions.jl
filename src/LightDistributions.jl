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


logpdf(d::Distribution, x) = logpdf(typeof(d))(x, params(d)...)
logpdf(::Type{<:Categorical}) = categorical_logpdf
logpdf(::Type{<:Poisson}) = poisson_logpdf
logpdf(::Type{<:Uniform}) = uniform_logpdf
logpdf(::Type{<:Normal}) = normal_logpdf
logpdf(::Type{<:Exponential}) = exponential_logpdf
logpdf(::Type{<:Gamma}) = gamma_logpdf
logpdf(::Type{<:Dirichlet}) = dirichlet_logpdf

CuArrays.@cufunc categorical_logpdf(args...) = _categorical_logpdf(args...)
CuArrays.@cufunc poisson_logpdf(args...) = _poisson_logpdf(args...)
CuArrays.@cufunc uniform_logpdf(args...) = _uniform_logpdf(args...)
CuArrays.@cufunc normal_logpdf(args...) = _normal_logpdf(args...)
CuArrays.@cufunc exponential_logpdf(args...) = _exponential_logpdf(args...)
CuArrays.@cufunc gamma_logpdf(args...) = _gamma_logpdf(args...)
CuArrays.@cufunc dirichlet_logpdf(args...) = _dirichlet_logpdf(args...)


end # module
