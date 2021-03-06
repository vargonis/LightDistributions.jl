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

@cufunc function lgamma(x::T) where T<:Real
    s = zero(T)
    for k in 9:-1:2
        s += T(_lgamma_cs[k]) / (x + k - 2)
    end
    s += T(_lgamma_cs[1])
    t = x + T(13)/2
    T(log2π)/2 + (x-one(T)/2)log(t) - t + log(s)
end
