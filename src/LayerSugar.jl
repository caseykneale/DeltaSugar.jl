function L1!( d::SugarLayer, λ )
    push!( d.weight_hooks, L1Hook( λ ) )
    return d
end

function L1( d::S, λ ) where { S <: SugarLayer } 
    weight_hooks = vcat(d.weight_hooks, L1Hook( λ ) )
    return S( copy(d.weight), weight_hooks, copy(d.bias), d.bias_hooks, d.σ )
end

L1( d::Flux.Dense, λ )  = DenseLayer( d.weight, [ L1Hook( λ ) ], d.bias, [], d.σ ) 
L1( c::Flux.Chain, λ)  = Chain( [ L1( l, λ ) for l in c ]... )

function L2!( d::SugarLayer, λ )
    push!( d.weight_hooks, L2Hook( λ ) )
    return d
end

function L2( d::S, λ ) where { S <: SugarLayer } 
    weight_hooks = vcat(d.weight_hooks, L2Hook( λ ) )
    return S( copy(d.weight), weight_hooks, copy(d.bias), d.bias_hooks, d.σ )
end

L2( d::Flux.Dense, λ )  = DenseLayer( d.weight, [ L2Hook( λ ) ], d.bias, [], d.σ ) 
L2( c::Flux.Chain, λ)  = Chain( [ L2( l, λ ) for l in c ]... )

function RevGrad!( d::SugarLayer) 
    push!(d.weight_hooks, ReverseGradientHook() )
    push!(d.bias_hooks, ReverseGradientHook() )
    return d
end

function RevGrad( d::S ) where { S <: SugarLayer } 
    weight_hooks = vcat(d.weight_hooks, ReverseGradientHook() )
    bias_hooks = vcat(d.bias_hooks, ReverseGradientHook() )
    return S( copy(d.weight), weight_hooks, copy(d.bias), bias_hooks, d.σ )
end

RevGrad( d::Flux.Dense)  = DenseLayer( d.weight, [ReverseGradientHook()], d.bias, [ReverseGradientHook()], d.σ ) 
RevGrad( c::Flux.Chain)  = Chain( [ RevGrad( l ) for l in c ]... ) 

function GradientClamp!( d::SugarLayer, ϵ )
    push!( d.weight_hooks, ClampHook( ϵ ) )
    push!( d.bias_hooks, ClampHook( ϵ ) )
    return d
end

function GradientClamp( d::S, ϵ ) where { S <: SugarLayer } 
    weight_hooks = vcat(d.weight_hooks, ClampHook( ϵ ) )
    bias_hooks = vcat(d.bias_hooks, ClampHook( ϵ ) )
    return S( copy(d.weight), weight_hooks, copy(d.bias), bias_hooks, d.σ )
end

GradientClamp( d::Flux.Dense, ϵ ) = DenseLayer( d.weight, [ ClampHook( ϵ ) ], d.bias, [ ClampHook( ϵ ) ], d.σ ) 
GradientClamp( c::Flux.Chain, ϵ ) = Chain( [ ClampHook( ϵ ) for l in c ]... )
