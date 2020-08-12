struct Normal{T<:Real} <: AbstractDistribution
    μ :: T
    σ :: T
    function Normal(μ::Real, σ::Real)
        T = promote_type(typeof(μ), typeof(σ))
        T <: Integer && (T = Float64)
        new{T}(μ, σ)
    end
end

@implement Distribution{Normal{T}, T} where T begin
    random(n::Normal) = randNormal(n.μ, n.σ)
    logpdf(::Type{<:Normal}) = logpdfNormal
end


randNormal(μ::T, σ::T) where T = μ + σ * randn(T)

@cufunc function logpdfNormal(x::Real, p::NamedTuple)
    x_, μ_, σ_ = promote(x, p.μ, p.σ)
    T = eltype(x_)
    iszero(σ_) && return ifelse(x_ == μ_, T(Inf), -T(Inf))
    -(((x_ - μ_) / σ_)^2 + T(log2π))/2 - log(σ_)
end
