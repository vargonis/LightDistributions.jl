module LightDistributions

using CanonicalTraits
using Random
using StatsFuns: poisinvcdf # Poisson
# using StaticArrays

using CUDAnative
using CuArrays
using MacroTools


# TODO extend Base.rand instead of defining own random
# import Base: rand

export Distribution, AbstractDistribution
export support, params, random, logpdf

_scalars = (:Categorical, :Poisson, :Uniform, :Normal, :Exponential, :Gamma)
_arrays = (:Dirichlet,)

for D in _scalars ∪ _arrays
    @eval export $D
end

export Mixture


############################
# Distribution generalities
############################

abstract type AbstractDistribution end

logpdf(d::AbstractDistribution) = x -> logpdf(typeof(d))(x, params(d)...)

function support(D::Type{<:AbstractDistribution})
    D.parameters[1] # fallback definition, distributions for which this is not adequate must override
end
support(d::AbstractDistribution) = support(typeof(d))

@trait Distribution{D, T} where {T = support(D)} begin
    params :: D => NamedTuple
    random :: D => T
    logpdf :: Type{D} => Function
end


###############################
# Distribution implementations
###############################

# function homogenize(t::Tuple)
#     T = promote_type(typeof(t).parameters...)
#     Tuple{Vararg{T}}(t)
# end

include("specfuns.jl")

for D in _scalars
    @eval include("scalars/" * $(String(D)) * ".jl")
end
include("arrays/Dirichlet.jl")
include("constructions.jl")


end # module
