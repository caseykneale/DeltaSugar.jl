struct TenseLayer{M,B,F} <: SugarLayer
    weight::M
    weight_hooks::Vector{GradientHook}
    bias::B
    bias_hooks::Vector{GradientHook}
    σ::F  
end

function TenseLayer(W::M, bias = true, σ::F = identity) where {M<:AbstractMatrix, F}
    b = Matrix(Flux.create_bias( W, bias, size( W, 2 ) )')
    TenseLayer( W, Vector{GradientHook}(undef,0), b, Vector{GradientHook}(undef,0), σ )
end

function TenseLayer(    in::Integer, out::Integer, σ = identity;
                        initW = nothing, initb = nothing,
                        init = Flux.glorot_uniform, bias = true )
    W = if initW !== nothing
        Base.depwarn("keyword initW is deprecated, please use init (which similarly accepts a funtion like randn)", :Dense)
        Flux.initW(in, out)
    else
        init(in, out)
    end
    b = if bias === true && initb !== nothing
        Base.depwarn("keyword initb is deprecated, please simply supply the bias vector, bias=initb(out)", :Dense)
        Flux.initb(out)
    else
        bias
    end
    return TenseLayer( W, b, σ )
end
Flux.@functor TenseLayer

function ( a::TenseLayer )( x::AbstractVecOrMat )
    W, b, σ = a.weight, a.bias, a.σ
    W_h = composer( a.weight_hooks )( W )
    b_h = composer( a.bias_hooks )( b )
    return σ.( x * W_h .+ b_h )
end

function Base.show(io::IO, l::TenseLayer)
    print(io, "TenseLayer(", size(l.weight, 2), ", ", size(l.weight, 1))
    l.σ == identity || print(io, ", ", l.σ)
    l.bias == Flux.Zeros() && print(io, ", bias=false")
    print(io, ")")
end