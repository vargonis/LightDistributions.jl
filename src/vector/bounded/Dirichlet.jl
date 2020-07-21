# Pasarse a FixedPointNumbers?
using StaticArrays


struct Dirichlet{N,T<:Real} <: AbstractDistribution
    α :: SVector{N,T}
    function Dirichlet(α::Tuple{Vararg{Real}})
        T = eltype(α)
        T <: Integer && (T = Float64)
        new{length(α),T}(α)
    end
end

Dirichlet(α...) = Dirichlet(α)
# Dirichlet(α::Real...) = Dirichlet(α)
# Dirichlet(α::Integer...) = Dirichlet(Float64.(α))
# Dirichlet(n::Integer, α::Real)= Dirichlet(ntuple(_->α, n))


support(::Type{<:Dirichlet{N,T}}) where {N,T} = SVector{N,T}

@implement Distribution{Dirichlet{N,T}, SVector{N,T}} where {N,T} begin
    params(d::Dirichlet) = (α=d.α,)
    random(d::Dirichlet) = randDirichlet(d.α)
    logpdf(::Type{<:Dirichlet}) = logpdfDirichlet
end


function randDirichlet(α::SVector{N,T}) where {N,T}
    # p = SVector(Tuple(randGamma(αi, one(T)) for αi in α))
    p = randGamma.(α, one(T))
    p / sum(p)
end
randDirichlet(α::Tuple{Vararg{Integer}}) = randDirichlet(Float64.(α))
randDirichlet(n::Integer, α::Real) = randDirichlet(Tuple(α for _ in 1:n))
randDirichlet(n::Integer, α::Integer) = randDirichlet(n, Float64(α))

@cufunc function logpdfDirichlet(x::SVector{N,<:Real}, α::SVector{N,<:Real}) where N
    # a, b = sum(u -> SVector(u,lgamma(u)), α)
    # s = sum(((u,v) -> (u-one(u))log(v)).(α, x))
    # s - b + lgamma(a)
    T = promote_type(eltype(x), eltype(α))
    a = b = s = zero(T)
    for (xi, αi) in zip(x, α)
        a += T(αi)
        b += lgamma(T(αi))
        s += (T(αi) - one(T))log(T(xi))
    end
    s - b + lgamma(a)
end
