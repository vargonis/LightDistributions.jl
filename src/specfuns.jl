using MacroTools

macro cufunc(ex)
    def = MacroTools.splitdef(ex)
    f = def[:name]
    def[:name] = Symbol(:cu, f)
    def[:body] = CuArrays.replace_device(def[:body])
    push!(CuArrays._cufuncs, f)
    quote
        $(esc(ex))
        $(esc(MacroTools.combinedef(def)))
        CuArrays.cufunc(::typeof($(esc(f)))) = $(esc(def[:name]))
    end
end


"""
logΓ function, computed via Lanczos approximation.
"""
function lgamma end

const log2π = log(2π)
const _lgamma_cs = (0.99999999999980993,
                    676.5203681218851,
                    -1259.1392167224028,
                    771.32342877765313,
                    -176.61502916214059,
                    12.507343278686905,
                    -0.13857109526572012,
                    9.9843695780195716e-6,
                    1.5056327351493116e-7)

@cufunc function lgamma(x::T) where T<:Real
    s = zero(T)
    for k in 9:-1:2
        s += T(_lgamma_cs[k]) / (x + k - 2)
    end
    s += T(_lgamma_cs[1])
    t = x + T(13)/2
    T(log2π)/2 + (x-one(T)/2)log(t) - t + log(s)
end


"""
Regularized incomplete beta function, computed from continued fraction expansion via
improved Lentz's algorithm.
"""
function cdfbeta end

@cufunc function cdfbeta(a::T, b::T, x::T) where T<:Real
    x ≤ zero(T) && return zero(T)
    x ≥ one(T) && return one(T)
    # There's good convergence for x < (a+1)/(a+b+2). Above this threshold, use symmetry:
    x > (a+1)/(a+b+2) && return 1 - cdfbeta(b,a,1-x)

    # Compute the coefficient in front of the continued fraction.
    # C = log(x)a + log1p(-x)b - log(a) - lgamma(a) - lgamma(b) + lgamma(a+b)
    C = x^a * (1-x)^b / a / exp(lgamma(a) + lgamma(b) - lgamma(a+b))

    # Use Lentz's algorithm to evaluate the continued fraction.
    r, c, d, k = one(T), one(T), T(Inf), 1
    while k ≤ 200
        f, c, d, k = _improved_lentz_iteration(c, d, k, a, b, x)
        r *= f
        # abs(1 - f) < T(1e-8) && return C - log(r)
        abs(1 - f) ≤ eps(T) && return C / r
    end
    throw(ProcessFailedException("Lentz algorithm didn't converge after 200 iterations"))
end

function _improved_lentz_iteration(c::T, d::T, k::Int, a::T, b::T, x::T) where T<:Real
    coef, k = _beta_continued_fraction_coef(k, a, b, x), k + 1
    f, c, d = 1, 1 + coef / c, 1 + coef / d # we assume c, d not tiny, so this is OK
    tiny_c, tiny_d = abs(c) ≤ 10eps(T), abs(d) ≤ 10eps(T)
    if tiny_c || tiny_d # don't want to return tinies, so use improved Lentz:
        # A partir de este punto, c y d almaceran productos parciales de ser necesario
        prev_c = prev_d = 1 # Necesario llevar la cuenta del ultimo producto parcial para calcular nuevos coeficientes al terminar la "mala racha" (seguidilla de tinies)
        while tiny_c || tiny_d
            coef, k = _beta_continued_fraction_coef(k, a, b, x), k + 1
            if tiny_c
                prev_c = c
                c += coef
                tiny_c = abs(c) ≤ 10eps(T)
            else
                # Terminó la mala racha para los c's, puedo hacer la actualizacion parcial correspondiente...
                f *= c
                # ...y dejar listo para continuar, puesto que quizas la mala racha sigue en los d's
                prev_c, c = 1, 1 + prev_c * coef / c
                tiny_c = abs(c) ≤ 10eps(T)
            end
            if tiny_d
                prev_d = d
                d += coef
                tiny_d = abs(d) ≤ 10eps(T)
            else
                f /= d
                prev_d, d = 1, 1 + prev_d * coef / d
                tiny_d = abs(d) ≤ 10eps(T)
            end
        end
    end
    # Al salir del while (de haber entrado), tendremos c,d adecuados pero faltara hacer una ultima actualizacion de f
    return f*c/d, c, d, k
end

function _beta_continued_fraction_coef(k::Int, a::Real, b::Real, x::Real)
    m, p = divrem(k, 2)
    p == 0 && return m * (b-m) / (a+k-1) / (a+k) * x
    return -(a+m) * (a+b+m) / (a+k-1) / (a+k) * x
end
