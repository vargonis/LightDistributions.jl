# Dirichlet
struct Dirichlet{P<:Tuple{Vararg{Real}}} <: Distribution{Tuple{Vararg{Real}},P}
    params :: P
    Dirichlet(; α) = new{typeof(α)}(α)
end

# Dirichlet(α::Real...) = Dirichlet(α)
# Dirichlet(α::Integer...) = Dirichlet(Float64.(α))
# Dirichlet(n::Integer, α::Real)= Dirichlet(ntuple(_->α, n))


function randDirichlet(α::Tuple{Vararg{T}}) where T<:Real
    p = Tuple(randGamma(αi, one(T)) for αi in α)
    p ./ sum(p)
end
randDirichlet(α::Tuple{Vararg{Integer}}) = randDirichlet(Float64.(α))
randDirichlet(n::Integer, α::Real) = randDirichlet(Tuple(α for _ in 1:n))
randDirichlet(n::Integer, α::Integer) = randDirichlet(n, Float64(α))

@cufunc function logpdfDirichlet(x::NTuple{N,Real}, α::NTuple{N,Real}) where N
    # a, b = sum(u -> SVector(u,lgamma(u)), α)
    # s = sum(((u,v) -> (u-one(u))log(v)).(α, x))
    # s - b + lgamma(a)
    T = promote_type(typeof.(x)..., typeof.(α)...)
    a = b = s = zero(T)
    for (xi, αi) in zip(x, α)
        a += T(αi)
        b += lgamma(T(αi))
        s += (T(αi) - one(T))log(T(xi))
    end
    s - b + lgamma(a)
end
