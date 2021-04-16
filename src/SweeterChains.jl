
function PlainChain( dims...; σ = identity )
    Flux.Chain( collect( Flux.Dense(i,f,σ) for (i,f) in zip( dims[1:(end-1)], dims[2:end] ) )...  )
end

function TChain( dims...; σ = identity )
    Flux.Chain( collect( TenseLayer(i,f,σ) for (i,f) in zip( dims[1:(end-1)], dims[2:end] ) )...  )
end
