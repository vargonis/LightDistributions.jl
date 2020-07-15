struct Mixture{Ds <: NTuple, W}
    weights :: W
    distributions :: Ds
end

support(::Type{Mixture{NTuple{N,D},W}}) where {N,D,W} = support(D)

@implement! Distribution{D,T} >: Distribution{Mixture{NTuple{N,D},W},T} where {N,D,W,T} begin
    params(m::Mixture) = (weights = m.weights, params = map(params, m.distributions))

    function random(m::Mixture)
        i = random(Categorical(m.weights))
        random(m.distributions[i])
    end

    function logpdf(::Type{Mixture{Ds}}) where Ds
        (x,ws,ps) -> sum(Tuple(w*logpdf(D)(x,p...) for (w,D,p) in zip(ws,Ds.parameters,ps)))
    end
end
