struct Mixture{Ds <: Tuple{Vararg{AbstractDistribution}},
               W <: Tuple{Vararg{Real}}} <: AbstractDistribution
    weights :: W
    distributions :: Ds
end

support(::Type{Mixture{NTuple{N,D},W}}) where {N,D,W} = support(D)

@implement Distribution{D,T} >: Distribution{Mixture{NTuple{N,D},W},T} where {N,D,W,T} begin
    params(m::Mixture) = (weights = m.weights, params = map(params, m.distributions))

    function random(m::Mixture)
        i = random(Categorical(m.weights))
        random(m.distributions[i])
    end

    function logpdf(::Type{Mixture{Ds}}) where Ds
        (x,ws,ps) -> sum(Tuple(w*logpdf(d)(x,p...) for (w,d,p) in zip(ws,Ds.parameters,ps)))
    end
end


struct Product{Ds <: Tuple{Vararg{AbstractDistribution}}} <: AbstractDistribution
    distributions :: Ds
end

Product(ds::AbstractDistribution...) = Product(ds)
support(::Type{Product{Ds}}) where Ds = Tuple{map(support, Ds.parameters)...}

# @implement Distribution{Product{Ds},T} where {T = support(Product{Ds})} begin
#     params(p::Product) = (params = map(params, p.distributions),)
#     random(p::Product) = map(random, p.distributions)
#     logpdf(::Type{Product{Ds}}) where Ds = x -> sum(D->logpdf(D)(x), Ds.parameters)
# end
@implement Distribution{D,T} >: Distribution{Product{NTuple{N,D}},NTuple{N,T}} where {N,D,T} begin
    params(p::Product) = (params = map(params, p.distributions),)
    random(p::Product) = map(random, p.distributions)
    logpdf(::Type{Product{Ds}}) where Ds = xs -> sum((D,x)->logpdf(D)(x), zip(Ds.parameters,xs))
end

# struct Product{D1<:AbstractDistribution, D2<:AbstractDistribution} <: AbstractDistribution
#     distributions :: Tuple{D1,D2}
# end
#
# support(::Type{Product{D1,D2}}) where {D1,D2} = Tuple{support(D1), support(D2)...}
#
# @implement Distribution{D1,T1} >: Distribution{D2,T2} >: Distribution{Product{D1,D2},Tuple{T1,T2}} begin
#     params(p::Product) = (params = map(params, p.distributions),)
#     random(p::Product) = map(random, p.distributions)
#     logpdf(::Type{Product{Ds}}) where Ds = x -> sum(D->logpdf(D)(x), Ds.parameters)
# end
