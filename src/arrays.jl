# Dirichlet
struct Dirichlet{N, F<:Real, T<:AbstractVector{F}, P<:AbstractVector{F}} <: Distribution{T,P}
    params :: P
    function Dirichlet{T}(params::AbstractVector) where T
        isconcretetype(T) || error("Dirichlet: parameter T must be concrete")
        T <: SVector && length(T) == length(params) || error("Dirichlet: inconsistent type and parameter length")
        new{length(params), eltype(T), T, typeof(params)}(params)
    end
end

Dirichlet{N}(α::Real) where N = Dirichlet{SVector{N,typeof(α)}}(SVector(ntuple(_->α, N)))


function rand(d::Dirichlet{N,F,T}) where {N,F,T}
    α = params(d)
    p = rand.(T([Gamma{F}(α[i],one(F)) for i in 1:N]))
    p / sum(p)
end

function logpdf(d::Dirichlet, x::AbstractVector{F}) where F
    α = F.(params(d))
    a, b = sum(u -> SVector(u,loggamma(u)), α)
    s = sum((u,v) -> (u-one(F)log(v)), zip(α,x))
    s - b + loggamma(a)
end

function _logpdf(d::Dirichlet, x::AbstractVector{F}) where F
    α = F.(params(d))
    a, b = sum(u -> SVector(u,CUDAnative.lgamma(u)), α)
    s = sum((u,v) -> (u-one(F)CUDAnative.log(v)), zip(α,x))
    s - b + CUDAnative.lgamma(a)
end
