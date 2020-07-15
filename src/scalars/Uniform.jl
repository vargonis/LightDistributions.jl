struct Uniform{T<:Real}
    a :: T
    b :: T
    function Uniform(a::Real, b::Real)
        T = promote_type(typeof(a), typeof(b))
        T <: Integer && (T = Float64)
        new{T}(a, b)
    end
end

@implement Distribution{Uniform{T}, T} where T begin
    params(u::Uniform) = (a=u.a, b=u.b)
    random(u::Uniform) = randUniform(u.a, u.b)
    logpdf(::Type{<:Uniform}) = logpdfUniform
end


randUniform(a::T, b::T) where T = a + (b - a)rand(T)

@cufunc function logpdfUniform(x_::Real, a_::Real, b_::Real)
    x, a, b = promote(x_, a_, b_)
    ifelse(a ≤ x ≤ b, -log(b - a), -typeof(x)(Inf))
end
