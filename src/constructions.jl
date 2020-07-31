# struct Mixture{Ds <: Tuple{Vararg{AbstractDistribution}},
#                W <: Tuple{Vararg{Real}}} <: AbstractDistribution
struct Mixture{Ds <: AbstractVector,
               W <: AbstractVector} <: AbstractDistribution
    weights :: W
    distributions :: Ds
end

support(::Type{<:Mixture{<:AbstractVector{D},W}}) where {D,W} = support(D)
Base.length(m::Mixture) = length(m.weights)

@implement Distribution{D,T} >: Distribution{Mixture{Ds,W},T} where {D,Ds<:AbstractVector{D},W,T} begin
    function params(m::Mixture)
        NamedTuple{(:weight,params(D)...)}.([(w,params(d)...) for (w,d) in zip(m.weights,m.distributions)])
    end

    function random(m::Mixture)
        i = random(Categorical(m.weights))
        random(m.distributions[i])
    end

    function logpdf(::Type{<:Mixture{Ds,W}})
        (x,ps) -> sum(p.weight*logpdf(D)(x,p) for p in ps)
    end
end


# Arrays
support(::Type{<:Array{D,N}}) where {D,N} = Array{support(D),N}

@implement Distribution{D,T} >: Distribution{Array{D,N},Array{T,N}} where {D,T,N} begin
    params(a::Array) = params.(a)
    random(a::Array) = random.(a)
    logpdf(::Type{<:Array{D,N}}) = (x,p) -> sum(logpdf(D).(x,p))
end


# Tuples
support(::Type{<:NTuple{N,D}}) where {N,D} = NTuple{N,support(D)}

@implement Distribution{D,T} >: Distribution{NTuple{N,D},NTuple{N,T}} where {D,T,N} begin
    params(a::NTuple) = params.(a)
    random(a::NTuple) = random.(a)
    logpdf(::Type{<:NTuple{N,D}}) = (x,p) -> sum(logpdf(D).(x,p))
end

# # Better to reuse Arrays for this
# struct Product{Ds <: Tuple{Vararg{AbstractDistribution}}} <: AbstractDistribution
#     distributions :: Ds
# end
#
# Product(ds::AbstractDistribution...) = Product(ds)
# support(::Type{Product{Ds}}) where Ds = Tuple{map(support, Ds.parameters)...}
#
# # @implement Distribution{Product{Ds},T} where {T = support(Product{Ds})} begin
# #     params(p::Product) = (params = map(params, p.distributions),)
# #     random(p::Product) = map(random, p.distributions)
# #     logpdf(::Type{Product{Ds}}) where Ds = x -> sum(D->logpdf(D)(x), Ds.parameters)
# # end
# @implement Distribution{D,T} >: Distribution{Product{NTuple{N,D}},NTuple{N,T}} where {N,D,T} begin
#     params(p::Product) = (params = map(params, p.distributions),)
#     random(p::Product) = map(random, p.distributions)
#     logpdf(::Type{Product{Ds}}) where Ds = xs -> sum((D,x)->logpdf(D)(x), zip(Ds.parameters,xs))
# end
#
# # struct Product{D1<:AbstractDistribution, D2<:AbstractDistribution} <: AbstractDistribution
# #     distributions :: Tuple{D1,D2}
# # end
# #
# # support(::Type{Product{D1,D2}}) where {D1,D2} = Tuple{support(D1), support(D2)...}
# #
# # @implement Distribution{D1,T1} >: Distribution{D2,T2} >: Distribution{Product{D1,D2},Tuple{T1,T2}} begin
# #     params(p::Product) = (params = map(params, p.distributions),)
# #     random(p::Product) = map(random, p.distributions)
# #     logpdf(::Type{Product{Ds}}) where Ds = x -> sum(D->logpdf(D)(x), Ds.parameters)
# # end
