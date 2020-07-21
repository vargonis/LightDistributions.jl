using StatsFuns: poisinvcdf


struct Poisson{T<:Real} <: AbstractDistribution
    λ :: T
end

support(::Type{<:Poisson}) = Int

@implement Distribution{Poisson{T}, Int} where T begin
    params(p::Poisson) = (λ=p.λ,)
    random(p::Poisson) = randPoisson(p.λ)
    logpdf(::Type{<:Poisson}) = logpdfPoisson
end

# TODO eliminar dependencia de StatsFuns:
randPoisson(λ::Real) = convert(Int, poisinvcdf(λ, rand()))

@cufunc function logpdfPoisson(i::Integer, λ::T) where T<:Real
    x = convert(eltype(T), i)
    iszero(λ) && return ifelse(iszero(x), zero(T), -T(Inf))
    x * log(λ) - λ - lgamma(x + one(T))
end
