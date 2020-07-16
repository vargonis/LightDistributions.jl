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
    params(n::NormalVector) = (μ=n.μ, σ=n.σ)
    random(n::NormalVector) = randNormalVector(n.μ, n.σ)
    logpdf(::Type{<:NormalVector}) = logpdfNormalVector
end


randNormalVector(μ, σ) = μ + σ * randn(eltype(μ), length(μ))

@cufunc function logpdfNormalVector(x, m, s) # cannot use σ because the macro would transform that to CuArrays.cufunc(σ)
    T = promote_type(eltype(x), eltype(m), eltype(s))
    t = s^(-1)
    v = t * (x - m)
    -(v⋅v + length(x)T(log2π))/2 - log(det(s))
end
