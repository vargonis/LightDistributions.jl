module LightDistributions

using CanonicalTraits
using Random
using CUDAnative
using CuArrays


# TODO extend Base.rand instead of defining own random
# import Base: rand

export Distribution, AbstractDistribution
export support, params, random, logpdf

_distributions = Dict(
    :Categorical  => "scalar/discrete/bounded/Categorical.jl",
    :Poisson      => "scalar/discrete/Poisson.jl",
    :Uniform      => "scalar/continuous/bounded/Uniform.jl",
    :Beta         => "scalar/continuous/bounded/Beta.jl",
    :Normal       => "scalar/continuous/Normal.jl",
    :Exponential  => "scalar/continuous/Exponential.jl",
    :Gamma        => "scalar/continuous/Gamma.jl",
    :Dirichlet    => "vector/bounded/Dirichlet.jl",
    :NormalVector => "vector/NormalVector.jl",
)

for D in keys(_distributions)
    @eval export $D
end

export Mixture, Product


############################
# Distribution generalities
############################

abstract type AbstractDistribution end

logpdf(d::Union{AbstractDistribution,AbstractArray}) = x -> logpdf(typeof(d))(x, params(d))

# Not sure if it's a good idea to have this default:
function support(D::Type{<:AbstractDistribution})
    D.parameters[1] # fallback definition, distributions for which this is not adequate must override
end
support(d::AbstractDistribution) = support(typeof(d))
params(::Type{D}) where D<:AbstractDistribution = fieldnames(D)

@trait Distribution{D, T} where {T = support(D)} begin
    params :: D => Any
    random :: D => T
    logpdf :: Type{D} => Function
    # Default implementation, declares all fields to be parameters:
    params(d::D) = NamedTuple{fieldnames(D)}(Tuple(getproperty(d,p) for p in fieldnames(D)))
end


###############################
# Distribution implementations
###############################

# function homogenize(t::Tuple)
#     T = promote_type(typeof(t).parameters...)
#     Tuple{Vararg{T}}(t)
# end

include("specfuns.jl")

for (_, path) in _distributions
    @eval include($path)
end
include("constructions.jl")


end # module
