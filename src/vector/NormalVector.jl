using LinearAlgebra
using StaticArrays


struct NormalVector{N, T<:Real} <: AbstractDistribution
    μ :: SVector{N,T}
    σ :: LowerTriangular{T,<:SMatrix{N,N,T}}
    function NormalVector(μ::AbstractVector{<:Real}, σ::AbstractMatrix{<:Real})
        N = length(μ)
        T = promote_type(eltype(μ), eltype(σ))
        T <: Integer && (T = Float64)
        μ isa SVector{N,T} || (μ = SVector{N}(T.(μ)))
        σ isa Union{LowerTriangular{<:SMatrix{N,N,T}},SMatrix{N,N,T}} || (σ = SMatrix{N,N}(T.(σ)))
        new{N,T}(μ, LowerTriangular(σ))
    end
end

support(::Type{NormalVector{N,T}}) where {N,T} = SVector{N,T}

@implement Distribution{NormalVector{N,T}, SVector{N,T}} where {N,T} begin
    random(n::NormalVector) = randNormalVector(n.μ, n.σ)
    logpdf(::Type{<:NormalVector}) = logpdfNormalVector
end


randNormalVector(μ, σ) = μ + σ * randn(eltype(μ), length(μ))

@cufunc function logpdfNormalVector(x, p::NamedTuple)
    T = promote_type(eltype(x), eltype(p.μ), eltype(p.σ))
    t = p.σ^(-1)
    v = t * (x - p.μ)
    -(v⋅v + length(x)T(log2π))/2 - log(det(p.σ))
end
