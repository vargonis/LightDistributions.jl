# Pasarse a FixedPointNumbers?
using StaticArrays


struct Dirichlet{T<:Real, A<:AbstractVector{T}} <: AbstractDistribution
    α :: A
end

Dirichlet(α::AbstractVector{T}) where T<:Integer = Dirichlet(Float64.(α))
Dirichlet(α::Real...) = Dirichlet(SVector{length(α)}(α))
# Dirichlet(α::Integer...) = Dirichlet(Float64.(α))
# Dirichlet(n::Integer, α::Real)= Dirichlet(ntuple(_->α, n))


support(::Type{<:Dirichlet{T,A}}) where {T,A} = A

@implement Distribution{Dirichlet{T,A}, A} where {T,A} begin
    random(d::Dirichlet) = randDirichlet(d.α)
    logpdf(::Type{<:Dirichlet}) = logpdfDirichlet
end


function randDirichlet(α::AbstractVector{T}) where T
    # p = SVector(Tuple(randGamma(αi, one(T)) for αi in α))
    p = randGamma.(α, one(T))
    p / sum(p)
end
# randDirichlet(α::Tuple{Vararg{Integer}}) = randDirichlet(Float64.(α))
# randDirichlet(n::Integer, α::Real) = randDirichlet(Tuple(α for _ in 1:n))
# randDirichlet(n::Integer, α::Integer) = randDirichlet(n, Float64(α))

@cufunc function logpdfDirichlet(x::AbstractVector{T}, p::NamedTuple) where T<:Real
    # a, b = sum(u -> SVector(u,lgamma(u)), α)
    # s = sum(((u,v) -> (u-one(u))log(v)).(α, x))
    # s - b + lgamma(a)
    T_ = promote_type(T, eltype(p.α))
    a = b = s = zero(T_)
    for (xi, αi) in zip(x, p.α)
        a += T_(αi)
        b += lgamma(T_(αi))
        s += (T_(αi) - one(T_))log(T_(xi))
    end
    s - b + lgamma(a)
end
