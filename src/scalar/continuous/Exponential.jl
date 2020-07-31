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

@cufunc function logpdfExponential(x::Real, p::NamedTuple)
    x_, λ_ = promote(x, p.λ)
    ifelse(x_ < zero(x_), -eltype(x_)(Inf), log(λ_) - λ_ * x_)
end
