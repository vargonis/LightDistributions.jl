# Dirichlet
struct Dirichlet{P<:Tuple{Vararg{Real}}} <: Distribution{AbstractVector{Real},P}
    params :: P
    Dirichlet(; α) = new{typeof(α)}(α)
end

# Dirichlet(α::Real...) = Dirichlet(α)
# Dirichlet(α::Integer...) = Dirichlet(Float64.(α))
# Dirichlet(n::Integer, α::Real)= Dirichlet(ntuple(_->α, n))


function randDirichlet(α::T...) where T<:Real
    p = SVector((αi -> randGamma(αi,one(T))).(α))
    p ./ sum(p)
end
randDirichlet(α::Integer...) = randDirichlet(Float64.(α)...)
randDirichlet(n::Integer, α::Real) = randDirichlet((α for _ in 1:n)...)
randDirichlet(n::Integer, α::Integer) = randDirichlet(n, Float64(α))

function logpdfDirichlet(x, α::Real...)
    a, b = sum(u -> SVector(u,loggamma(u)), α)
    s = sum(((u,v) -> (u-one(u))log(v)).(α, x))
    s - b + lgamma(a)
end

function _logpdfDirichlet(x, α::Real...)
    a, b = sum(u -> SVector(u,lgamma(u)), α)
    s = sum(((u,v) -> (u-one(u))CUDAnative.log(v)).(α, x))
    s - b + CUDAnative.lgamma(a)
end
