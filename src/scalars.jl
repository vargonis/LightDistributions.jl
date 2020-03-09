# Base.@irrational log2π 1.8378770664093454836 log(big(2.)*π)

function homogenize(t::Tuple)
    T = promote_type(typeof(t).parameters...)
    Tuple{Vararg{T}}(t)
end


# Discrete distributions
for (D, P) in [(:Categorical, Tuple{Vararg{Real}}),
               (:Poisson, Tuple{Real})]
    @eval struct $D{T<:Integer, P<:$P} <: Distribution{T,P}
        params :: P
    end
    @eval function $D(ps...)
        params = homogenize(ps)
        $D{Int64, typeof(params)}(params)
    end
end


# Continuous distributions
for (D, P) in [(:Uniform, Tuple{Real,Real}),
               (:Normal, Tuple{Real,Real}),
               (:Exponential, Tuple{Real}),
               (:Gamma, Tuple{Real,Real})]
    @eval struct $D{T<:Real, P<:$P} <: Distribution{T,P}
        params :: P
    end
    @eval function $D(ps...)
        params = homogenize(ps)
        T = eltype(params)
        $D{T, typeof(params)}(params)
    end
end


##########################
# Sampling and densities
##########################

# Categorical
function rand(d::Categorical{I}) where I<:Integer
    p = params(d)
    draw = rand(eltype(p))
    cp = zero(eltype(p))
    i = zero(I)
    while cp < draw && i < length(p)
        cp += p[i+=1]
    end
    max(i, one(I))
end

categorical_logpdf(i::I, p::Real...) where I =
    ifelse(one(I) ≤ i ≤ length(p), @inbounds log(p[i]), -eltype(p)(Inf))

_categorical_logpdf(i::I, p::Real...) where I =
    ifelse(one(I) ≤ i ≤ length(p), @inbounds CUDAnative.log(p[i]), -eltype(p)(Inf))


# Poisson
# TODO eliminar dependencia de StatsFuns:
function rand(d::Poisson{I}) where I
    λ = params(d)
    convert(I, poisinvcdf(λ, rand()))
end

poisson_logpdf(i::Integer, λ::Real) = poislogpdf(λ, i)

function _poisson_logpdf(i::Integer, λ::T) where T<:Real
    x = convert(T, i)
    iszero(λ) && return ifelse(iszero(x), zero(T), -T(Inf))
    x * CUDAnative.log(λ) - λ - CUDAnative.lgamma(x + one(T))
end


# Uniform
function rand(d::Uniform{T}) where T
    a::T, b::T = params(d)
    a + (b - a)rand(T)
end

uniform_logpdf(x, a::T, b::T) where T =
    ifelse(a ≤ T(x) ≤ b, -log(b - a), -T(Inf))

_uniform_logpdf(x, a::T, b::T) where T =
    ifelse(a ≤ T(x) ≤ b, -CUDAnative.log(b - a), -T(Inf))


# Normal
function rand(d::Normal{T}) where T
    μ::T, σ::T = params(d)
    μ + σ * randn(T)
end

function normal_logpdf(x, μ::T, σ::T) where T
    iszero(σ) && return ifelse(T(x) == μ, T(Inf), -T(Inf))
    -(((T(x) - μ) / σ)^2 + T(log2π))/2 - log(σ)
end

function _normal_logpdf(x, μ::T, σ::T) where T
    iszero(σ) && return ifelse(T(x) == μ, T(Inf), -T(Inf))
    -(((T(x) - μ) / σ)^2 + T(log2π))/2 - CUDAnative.log(σ)
end


# Exponential
function rand(d::Exponential{T}) where T
    λ::T = params(d)
    λ * randexp(T)
end

exponential_logpdf(x, λ::T) where T =
    ifelse(x < zero(x), -T(Inf), log(λ) - λ * T(x))

_exponential_logpdf(x, λ::T) where T =
    ifelse(x < zero(x), -T(Inf), CUDAnative.log(λ) - λ * T(x))


# Gamma
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

function rand(d::Gamma{T}) where T
    α::T, θ::T = params(d)
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

# TODO eliminar dependencia de StatsFuns:
gamma_logpdf(x, α::T, θ::T) where T =
    gammalogpdf(α, θ, x)

_gamma_logpdf(x, α::T, θ::T) where T =
    -CUDAnative.lgamma(α) - α*CUDAnative.log(θ) + (α-one(T))CUDAnative.log(T(x)) - T(x)/θ
