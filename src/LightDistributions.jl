module LightDistributions

using Random
using StatsFuns: poisinvcdf
using StaticArrays
using CUDAnative
using CuArrays
using MacroTools


# TODO extend Base.rand instead of defining own random
# import Base: rand

include("specfuns.jl")


############################
# Distribution generalities
############################

const _distributions = (
    :Categorical, :Poisson,
    :Uniform, :Normal, :Exponential, :Gamma,
    :Dirichlet
)
const _params = (
    Categorical = NamedTuple{(:p,), Tuple{Tuple{Vararg{Real}}}},
    Poisson     = NamedTuple{(:λ,), Tuple{Real}},
    Uniform     = NamedTuple{(:a,:b), Tuple{Real,Real}},
    Normal      = NamedTuple{(:μ,:σ), Tuple{Real,Real}},
    Exponential = NamedTuple{(:λ,), Tuple{Real}},
    Gamma       = NamedTuple{(:k,:θ), Tuple{Real,Real}},
    Dirichlet   = NamedTuple{(:α,), Tuple{Tuple{Vararg{Real}}}}
)

export Distribution
export random, params, logpdf
for s in _distributions
    @eval export $s
end
export Mixture

"""
An abstract type for a distribution with parameters of type P and values of type T.
"""
abstract type Distribution end # Dropped parameter T because invariance means, for instance, that Distribution{Float64} <: Distribution{Real} does not hold; thus, it feels irrelevant.

# Base.eltype(::Distribution{T}) where T = T
# Base.eltype(::Type{<:Distribution{T}}) where T = T

params(d::Distribution) = d.params

###############################
# Distribution implementations
###############################

# function homogenize(t::Tuple)
#     T = promote_type(typeof(t).parameters...)
#     Tuple{Vararg{T}}(t)
# end
#
# function homogenize(s::Tuple, t::Tuple)
#     T = promote_type(typeof(t).parameters..., typeof(t).parameters...)
#     Tuple{Vararg{T}}(s), Tuple{Vararg{T}}(t)
# end

include("scalars.jl")
include("arrays.jl")
include("constructions.jl")

for D in _distributions
    @eval params(::Type{<:$D}) = $(_params[D])
    @eval logpdf(::Type{<:$D}) = $(Symbol(:logpdf,D))
    @eval logpdf(d::$D) = x -> $(Symbol(:logpdf,D))(x, d.params...)
    @eval random(::Type{<:$D}) = $(Symbol(:rand, D))
    @eval random(d::$D) = $(Symbol(:rand, D))(d.params...)
end


end # module
