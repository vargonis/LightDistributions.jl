# Discrete distributions
for D in [:Categorical, :Poisson]
    @eval struct $D{P<:$(_params[D].parameters[2])} <: Distribution #{Integer}
        params :: P
        function $D(; kwargs...)
            params = Tuple(values(kwargs))
            new{typeof(params)}(params)
        end
    end
end

# Continuous distributions
for D in [:Uniform, :Normal, :Exponential, :Gamma]
    @eval struct $D{P<:$(_params[D].parameters[2])} <: Distribution #{Real}
        params :: P
        function $D(; kwargs...)
            params = Tuple(values(kwargs))
            new{typeof(params)}(params)
        end
    end
end


##########################
# Sampling and densities
##########################

# Categorical
function randCategorical(p::Tuple{Vararg{T}}) where T<:Real
    draw = rand(T)
    cp = zero(T)
    i = zero(Int)
    while cp < draw && i < length(p)
        cp += p[i+=1]
    end
    max(i, 1)
end
randCategorical(p::T...) where T<:Integer = randCategorical(Float64.(p)...)


@cufunc function logpdfCategorical(i::I, p::Tuple{Vararg{Real}}) where I<:Integer
    # p = homogenize(p_) # a bit wasteful, only need to convert p[i]
    # T = eltype(eltype(p))
    T = promote_type(eltype.(p)...)
    ifelse(1 ≤ i ≤ length(p), @inbounds log(T(p[i])), -T(Inf))
end


# Poisson
# TODO eliminar dependencia de StatsFuns:
randPoisson(λ::Real) = convert(Int, poisinvcdf(λ, rand()))

@cufunc function logpdfPoisson(i::Integer, λ::T) where T<:Real
    x = convert(eltype(T), i)
    iszero(λ) && return ifelse(iszero(x), zero(T), -T(Inf))
    x * log(λ) - λ - lgamma(x + one(T))
end


# Uniform
randUniform(a::T, b::T) where T<:Real = a + (b - a)rand(T)
randUniform(a::T, b::T) where T<:Integer = Float64(a) + Float64(b - a)rand()

@cufunc function logpdfUniform(x_::Real, a_::Real, b_::Real)
    x, a, b = promote(x_, a_, b_)
    ifelse(a ≤ x ≤ b, -log(b - a), -typeof(x)(Inf))
end


# Normal
randNormal(μ::T, σ::T) where T<:Real = μ + σ * randn(T)
randNormal(μ::T, σ::T) where T<:Integer = Float64(μ) + Float64(σ) * randn()

@cufunc function logpdfNormal(x_::Real, μ_::Real, σ_::Real)
    x, m, s = promote(x_, μ_, σ_) # cannot use σ because the macro would transform that to CuArrays.cufunc(σ)
    T = eltype(x)
    iszero(s) && return ifelse(x == m, T(Inf), -T(Inf))
    -(((x - m) / s)^2 + T(log2π))/2 - log(s)
end


# Exponential
randExponential(λ::Real) = λ * randexp(eltype(λ))
randExponential(λ::Integer) = λ * randexp()

@cufunc function logpdfExponential(x_::Real, λ_::Real)
    x, λ = promote(x_, λ_)
    ifelse(x < zero(x), -eltype(x)(Inf), log(λ) - λ * x)
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

@cufunc function logpdfGamma(x_::Real, α_::Real, θ_::Real)
    x, α, θ = promote(x_, α_, θ_)
    T = eltype(x)
    -lgamma(α) - α*log(θ) + (α-one(T))log(x) - x/θ
end
