struct Mixture{Ds <: AbstractVector,
               W <: AbstractVector} <: AbstractDistribution
    weights :: W
    distributions :: Ds
end

support(::Type{<:Mixture{<:AbstractVector{D},W}}) where {D,W} = support(D)
Base.length(m::Mixture) = length(m.weights)

@implement Distribution{D,T} >: Distribution{Mixture{Ds,W},T} where {D,Ds<:AbstractVector{D},W,T} begin
    # Seria bonito pero no funciona en general:
    # function params(m::Mixture)
    #     NamedTuple{(:weight,params(D)...)}.([(w,params(d)...) for (w,d) in zip(m.weights,m.distributions)])
    # end
    params(m::Mixture) = (weights = m.weights, params = map(params, m.distributions))

    function random(m::Mixture)
        i = random(Categorical(m.weights))
        random(m.distributions[i])
    end

    function logpdf(::Type{<:Mixture{Ds,W}})
        (x,p) -> sum(w*logpdf(D)(x,q) for (w,q) in zip(p.weights,p.params))
    end
end


# Arrays
support(::Type{<:Array{D,N}}) where {D,N} = Array{support(D),N}
support(::Type{<:SArray{S,D,N,L}}) where {S,D,N,L} = SArray{S,support(D),N,L}

@implement Distribution{D,T} >: Distribution{Array{D,N},Array{T,N}} where {D,T,N} begin
    params(a::Array) = params.(a)
    random(a::Array) = random.(a)
    logpdf(::Type{<:Array{D,N}}) = (x,p) -> sum(logpdf(D).(x,p))
end

@implement Distribution{D,T} >: Distribution{SArray{S,D,N,L},SArray{S,T,N,L}} where {D,T,S,N,L} begin
    params(a::SArray) = params.(a)
    random(a::SArray) = random.(a)
    logpdf(::Type{<:SArray{S,D,N,L}}) = (x,p) -> sum(logpdf(D).(x,p))
end


# Tuples
support(::Type{<:NTuple{N,D}}) where {N,D} = NTuple{N,support(D)}

@implement Distribution{D,T} >: Distribution{NTuple{N,D},NTuple{N,T}} where {D,T,N} begin
    params(a::NTuple) = params.(a)
    random(a::NTuple) = random.(a)
    logpdf(::Type{<:NTuple{N,D}}) = (x,p) -> sum(logpdf(D).(x,p))
end
