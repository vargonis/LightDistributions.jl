struct Categorical{N,T<:Real} <: AbstractDistribution
    p :: NTuple{N,T}
end
Categorical(p...) = Categorical(p)

support(::Type{<:Categorical}) = Int

@implement Distribution{Categorical{N,T}, Int} where {N,T} begin
    params(c::Categorical) = (p=c.p,)
    random(c::Categorical) = randCategorical(c.p)
    logpdf(::Type{<:Categorical}) = logpdfCategorical
end

function randCategorical(p::Tuple{Vararg{T}}) where T<:Real
    draw = rand(T)
    cp = zero(T)
    i = zero(Int)
    while cp < draw && i < length(p)
        cp += p[i+=1]
    end
    max(i, 1)
end
# randCategorical(p::T...) where T<:Integer = randCategorical(Float64.(p)...)

@cufunc function logpdfCategorical(i::I, p::Tuple{Vararg{Real}}) where I<:Integer
    T = promote_type(eltype.(p)...)
    ifelse(1 ≤ i ≤ length(p), @inbounds log(T(p[i])), -T(Inf))
end
