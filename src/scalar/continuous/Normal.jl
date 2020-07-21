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
    params(n::Normal) = (μ=n.μ, σ=n.σ)
    random(n::Normal) = randNormal(n.μ, n.σ)
    logpdf(::Type{<:Normal}) = logpdfNormal
end


randNormal(μ::T, σ::T) where T = μ + σ * randn(T)

@cufunc function logpdfNormal(x_::Real, μ_::Real, σ_::Real)
    x, m, s = promote(x_, μ_, σ_) # cannot use σ because the macro would transform that to CuArrays.cufunc(σ)
    T = eltype(x)
    iszero(s) && return ifelse(x == m, T(Inf), -T(Inf))
    -(((x - m) / s)^2 + T(log2π))/2 - log(s)
end
