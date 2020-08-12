struct Uniform{T<:Real} <: AbstractDistribution
    a :: T
    b :: T
    function Uniform(a::Real, b::Real)
        T = promote_type(typeof(a), typeof(b))
        T <: Integer && (T = Float64)
        new{T}(a, b)
    end
end

@implement Distribution{Uniform{T}, T} where T begin
    random(u::Uniform) = randUniform(u.a, u.b)
    logpdf(::Type{<:Uniform}) = logpdfUniform
end


randUniform(a::T, b::T) where T = a + (b - a)rand(T)

@cufunc function logpdfUniform(x::Real, p::NamedTuple)
    x_, a_, b_ = promote(x, p.a, p.b)
    ifelse(a_ ≤ x_ ≤ b_, -log(b_ - a_), -typeof(x_)(Inf))
end
