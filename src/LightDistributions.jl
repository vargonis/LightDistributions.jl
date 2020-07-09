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
export sample, random, params, logpdf
for s in _distributions
    @eval export $s
end

abstract type Distribution{T,P} end

Base.eltype(::Distribution{T}) where T = T
Base.eltype(::Type{<:Distribution{T}}) where T = T

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

for D in _distributions
    @eval params(::Type{<:$D}) = $(_params[D])
    @eval logpdf(::Type{<:$D}) = $(Symbol(:logpdf,D)) # definirlo tb para instancias!
    @eval sample(::Type{<:$D}) = $(Symbol(:rand, D)) # se puede llamar rand igual...
    @eval random(d::$D) = $(Symbol(:rand, D))(d.params...)
end


end # module
