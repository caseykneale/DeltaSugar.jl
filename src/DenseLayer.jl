struct DenseLayer{M,B,F} <: SugarLayer
    weight::M
    weight_hooks::Vector{GradientHook}
    bias::B
    bias_hooks::Vector{GradientHook}
    σ::F 
end
function DenseLayer(W::M, bias = true, σ::F = identity) where {M<:AbstractMatrix, F}
    b = Flux.create_bias(W, bias, size(W,1))
    DenseLayer( W, Vector{GradientHook}(undef,0), b, Vector{GradientHook}(undef,0), σ)
end

function DenseLayer(    in::Integer, out::Integer, σ = identity;
                        initW = nothing, initb = nothing,
                        init = Flux.glorot_uniform, bias = true )
    W = if initW !== nothing
        Base.depwarn("keyword initW is deprecated, please use init (which similarly accepts a funtion like randn)", :Dense)
        Flux.initW(out, in)
    else
        init(out, in)
    end
    b = if bias === true && initb !== nothing
        Base.depwarn("keyword initb is deprecated, please simply supply the bias vector, bias=initb(out)", :Dense)
        Flux.initb(out)
    else
        bias
    end
    return DenseLayer( W, b, σ )
end
Flux.@functor DenseLayer

function ( a::DenseLayer )( x::AbstractVecOrMat )
    W, b, σ = a.weight, a.bias, a.σ
    W_h = composer( a.weight_hooks )( W )
    b_h = composer( a.bias_hooks )( b )
    return σ.( W_h * x .+ b_h )
end

function Base.show(io::IO, l::DenseLayer)
    print(io, "DenseLayer(", size(l.weight, 2), ", ", size(l.weight, 1))
    l.σ == identity || print(io, ", ", l.σ)
    l.bias == Flux.Zeros() && print(io, ", bias=false")
    print(io, ")")
end