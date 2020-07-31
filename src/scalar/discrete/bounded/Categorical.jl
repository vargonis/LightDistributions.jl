using StaticArrays

struct Categorical{P<:AbstractVector{<:Real}} <: AbstractDistribution
    p :: P # privilegiar SVector, por consistencia con Dirichlet
    # Categorical(p) = new{length(p),eltype(p)}(p)
end
Categorical(p::Real...) = Categorical(p)

support(::Type{<:Categorical}) = Int

@implement Distribution{Categorical{P}, Int} where P begin
    params(c::Categorical) = (p=c.p,)
    random(c::Categorical) = randCategorical(c.p)
    logpdf(::Type{<:Categorical}) = logpdfCategorical
end

function randCategorical(p::AbstractVector{T}) where T<:Real
    draw = rand(T)
    cp = zero(T)
    i = zero(Int)
    while cp < draw && i < length(p)
        cp += p[i+=1]
    end
    max(i, 1)
end
# randCategorical(p::T...) where T<:Integer = randCategorical(Float64.(p)...)

@cufunc function logpdfCategorical(i::I, p::NamedTuple) where I<:Integer
    T = promote_type(eltype.(p.p)...)
    ifelse(1 ≤ i ≤ length(p.p), @inbounds log(T(p.p[i])), -T(Inf))
end
