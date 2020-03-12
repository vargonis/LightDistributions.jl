# LightDistributions

This package is meant to provide basic infrastructure for probabilistic programming, with CuArrays support and Tracker/Zygote compatibility.


## A GPU example

Generate a normally distributed random variable and compute its likelihood with respect to several different mean parameters:
```julia
julia> x = randn()
3.022014755973123

julia> logpdf(Normal).(x, CuArray(collect(-10.:1.:10.)), 1.)
21-element CuArray{Float64,1,Nothing}:
 -85.70537268559555  
 -73.18335792962242  
 -61.6613431736493   
 -51.139328417676175
 -41.61731366170306  
 -33.09529890572993  
 -25.57328414975681  
 -19.051269393783688
 -13.529254637810565
  -9.007239881837442
   ⋮                 
  -1.441195613918074
  -0.9191808579449507
  -1.3971661019718276
  -2.8751513459987046
  -5.353136590025581
  -8.831121834052459
 -13.309107078079338
 -18.787092322106215
 -25.26507756613309  
```

## An AD example

Make a simple maximum likelihood estimation using Zygote:
```julia
Zygote.@adjoint logpdf(d) = logpdf(d), _ -> 0

n = 10000
data = CuArray([sample(Gamma)(.5,.9) for _ in 1:n])
μ, σ = rand(), rand()

ϵ = .01
for i in 1:n
    dμ, dσ = gradient(μ, σ) do μ, σ
        sum(logpdf(Gamma).(data, μ, σ)) / n
    end
    μ += ϵ * dμ; σ += ϵ * dσ
    i % 1000 == 0 && @show μ, σ
end
```
