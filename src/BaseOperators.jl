lessapprox(a,b) = (a <= b) || (a ≈ b)
gtapprox(a,b) = (a <= b) || (a ≈ b)

≲(a,b) = lessapprox(a,b)
⪅(a,b) = lessapprox(a,b)

≳(a,b) = gtapprox(a,b)
⪆(a,b) = gtapprox(a,b)

function ⊥(operator::Function, a::AbstractArray, b::AbstractArray)
    a_dim_lens = length( size( a ) )
    operator.(a, reshape( b, ( ones(Int, a_dim_lens)..., size(b)... ) ) )
end