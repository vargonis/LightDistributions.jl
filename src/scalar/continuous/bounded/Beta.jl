# using FixedPointNumbers

struct Beta{T<:Real} <: AbstractDistribution
    α :: T
    β :: T
    function Beta(α::Real, β::Real)
        T = promote_type(typeof(α), typeof(β))
        T <: Integer && (T = Float64)
        new{T}(α, β)
    end
end


@implement Distribution{Beta{T}, T} where T begin
    params(b::Beta) = (α=b.α, β=b.β)
    random(b::Beta) = randBeta(b.α, b.β)
    logpdf(::Type{<:Beta}) = logpdfBeta
end


function randBeta(α::T, β::T) where T
    p, q = randGamma.((α, β), one(T))
    p / (p + q)
end

@cufunc function logpdfBeta(x_::Real, α_::Real, β_::Real)
    x, α, β = promote(x_, α_, β_)
    T = eltype(x)
    (α - one(T))log(x) + (β - one(T))log1p(-x) - lgamma(α) - lgamma(β) + lgamma(α + β)
end
