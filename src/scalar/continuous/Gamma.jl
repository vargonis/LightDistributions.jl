struct Gamma{T<:Real} <: AbstractDistribution
    α :: T
    θ :: T
    function Gamma(α::Real, θ::Real)
        T = promote_type(typeof(α), typeof(θ))
        T <: Integer && (T = Float64)
        new{T}(α, θ)
    end
end

@implement Distribution{Gamma{T}, T} where T begin
    random(g::Gamma) = randGamma(g.α, g.θ)
    logpdf(::Type{<:Gamma}) = logpdfGamma
end


function _MarsagliaTsang2000(α::T) where T<:Real
    d = α - one(T)/3
    c = one(T) / sqrt(9d)
    while true
        x = randn(T)
        v = (one(T) + c*x)^3
        while v < zero(T)
            x = randn(T)
            v = (one(T) + c*x)^3
        end
        u = rand(T)
        # u < one(F) - F(0.0331)x^4 && return d*v
        log(u) < x^2/2 + d*(one(T) - v + log(v)) && return d*v
    end
end

function randGamma(α::T, θ::T) where T<:Real
    if α < one(T) # use the γ(1+α)*U^(1/α) trick from Marsaglia and Tsang (2000)
        x = θ * _MarsagliaTsang2000(α + one(T)) # Gamma(α + 1, θ)
        e = randexp(T)
        return x * exp(-e / α)
    elseif α == one(T)
        return θ * randexp(eltype(T))
    else
        return θ * _MarsagliaTsang2000(α)
    end
end
randGamma(α::T, θ::T) where T<:Integer = randGamma(Float64(α), Float64(θ))

@cufunc function logpdfGamma(x::Real, p::NamedTuple)
    x_, α_, θ_ = promote(x, p.α, p.θ)
    T = eltype(x_)
    -lgamma(α_) - α_*log(θ_) + (α_-one(T))log(x_) - x_/θ_
end
