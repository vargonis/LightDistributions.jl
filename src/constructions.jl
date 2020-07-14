struct Mixture{
            Ds <: Tuple{Vararg{Distribution}},
            Ws <: Tuple{Vararg{Real}},
        } <: Distribution
    distributions :: Ds
    weights :: Ws
    Mixture(ds::Tuple{Vararg{Distribution}}, ws::Tuple{Vararg{Real}}) =
        new{typeof(ds), typeof(ws)}(ds, ws)
end

params(m::Mixture) = (m.weights, map(params, m.distributions)...)

function random(m::Mixture)
    i = random(Categorical(p = m.weights))
    random(m.distributions[i])
end

function logpdf(::Type{Mixture{Ds}}) where Ds
    (x,ws,ps...) -> sum(Tuple(w*logpdf(D)(x,p...) for (w,D,p) in zip(ws,Ds.parameters,ps)))
end

function logpdf(m::Mixture)
    x -> logpdf(typeof(m))(x, params(m)...)
end
