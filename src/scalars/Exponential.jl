struct Exponential{T<:Real} <: AbstractDistribution
    λ :: T
    function Exponential(λ::Real)
        T = typeof(λ)
        T <: Integer && (T = Float64)
        new{T}(λ)
    end
end

@implement Distribution{Exponential{T}, T} where T begin
    params(e::Exponential) = (λ=e.λ,)
    random(e::Exponential) = randExponential(e.λ)
    logpdf(::Type{<:Exponential}) = logpdfExponential
end


randExponential(λ::T) where T = λ * randexp(T)

@cufunc function logpdfExponential(x_::Real, λ_::Real)
    x, λ = promote(x_, λ_)
    ifelse(x < zero(x), -eltype(x)(Inf), log(λ) - λ * x)
end
