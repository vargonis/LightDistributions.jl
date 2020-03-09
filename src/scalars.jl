# Base.@irrational log2π 1.8378770664093454836 log(big(2.)*π)


# Discrete distributions
for (D, P) in [(:Categorical, Tuple{Vararg{Real}}),
               (:Poisson, Tuple{Real})]
    @eval struct $D{T<:Integer, P} <: Distribution{T,P}
        params :: P
        $D{T}(params::$P) where T = new{T,typeof(params)}(params)
    end
    @eval $D{T}(params...) where T = $D{T}(params)
    # @eval $D(params...) =
end


# Continuous distributions
for (D, P) in [(:Uniform, Tuple{Real,Real}),
               (:Normal, Tuple{Real,Real}),
               (:Exponential, Tuple{Real}),
               (:Gamma, Tuple{Real,Real})]
    @eval struct $D{T<:Real, P} <: Distribution{T,P}
        params :: P
        $D{T}(params::$P) where T = new{T,typeof(params)}(params)
    end
    @eval $D{T}(params...) where T = $D{T}(params)
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

function logpdf(d::Categorical{I}, i::I) where I
    p = params(d)
    ifelse(one(I) ≤ i ≤ length(p), @inbounds log(p[i]), -typeof(p)(Inf))
end

function _logpdf(d::Categorical{I}, i::I) where I
    p = params(d)
    ifelse(one(I) ≤ i ≤ length(p), @inbounds CUDAnative.log(p[i]), -typeof(p)(Inf))
end


# Poisson
# TODO eliminar dependencia de StatsFuns:
function rand(d::Poisson{I}) where I
    λ = params(d)
    convert(I, poisinvcdf(λ, rand()))
end

logpdf(d::Poisson{I}, i::I) where I = poislogpdf(params(d), i)

function _logpdf(d::Poisson{I}, i::I) where I
    λ = params(d)
    T = typeof(λ)
    x = convert(T, i)
    iszero(λ) && return ifelse(iszero(x), zero(T), -T(Inf))
    x * CUDAnative.log(λ) - λ - CUDAnative.lgamma(x + one(T))
end


# Uniform
function rand(d::Uniform{T}) where T
    a::T, b::T = params(d)
    a + (b - a)rand(T)
end

function logpdf(d::Uniform{T}, x::T) where T
    a::T, b::T = params(d)
    ifelse(a ≤ x ≤ b, -log(b - a), -T(Inf))
end

function _logpdf(d::Uniform{T}, x::T) where T
    a::T, b::T = params(d)
    ifelse(a ≤ x ≤ b, -CUDAnative.log(b - a), -T(Inf))
end


# Normal
function rand(d::Normal{T}) where T
    μ::T, σ::T = params(d)
    μ + σ * randn(T)
end

function logpdf(d::Normal{T}, x) where T
    μ, σ = params(d)
    iszero(σ) && return ifelse(x == μ, T(Inf), -T(Inf))
    -(((x - μ) / σ)^2 + T(log2π))/2 - log(σ)
end

function _logpdf(d::Normal{T}, x) where T
    μ, σ = params(d)
    iszero(σ) && return ifelse(x == μ, T(Inf), -T(Inf))
    -(((x - μ) / σ)^2 + T(log2π))/2 - CUDAnative.log(σ)
end


# Exponential
function rand(d::Exponential{T}) where T
    λ::T = params(d)
    λ * randexp(T)
end

function logpdf(d::Exponential{T}, x::T) where T
    λ::T = params(d)
    ifelse(x < zero(T), -T(Inf), log(λ) - λ * x)
end

function _logpdf(d::Exponential{T}, x::T) where T
    λ::T = params(d)
    ifelse(x < zero(T), -T(Inf), CUDAnative.log(λ) - λ * x)
end


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
function logpdf(d::Gamma{T}, x::T) where T
    α::T, θ::T = params(d)
    gammalogpdf(α, θ, x)
end

function _logpdf(d::Gamma{T}, x::T) where T
    α::T, θ::T = params(d)
    -CUDAnative.lgamma(α) - α*CUDAnative.log(θ) + (α-one(T))*CUDAnative.log(x) - x/θ
end
