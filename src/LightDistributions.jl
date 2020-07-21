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

for (_, path) in _distributions
    @eval include($path)
end
include("constructions.jl")


end # module
