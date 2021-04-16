function composer(fns::Vector{GradientHook}) 
    return if length( fns ) == 0 
        identity 
    else
        reduce( ∘ , reverse( fns ) )
    end
end
composer(fns::Nothing) = identity

struct IdentityHook <: GradientHook; end
(a::IdentityHook)( x ) = x
@adjoint (a::IdentityHook)( x ) = x, Δ -> Δ
Base.show(io::IO, l::IdentityHook) = print(io, "Identity Gradient Transform")

struct L1Hook{F} <: GradientHook
    λ::F
end
ΔL1(Δ, θ, λ) = ( Δ .+ ( λ .* sign.( Δ ) ) ) .* ( abs.( θ ) .> λ )
( a::L1Hook )( x ) = x
@adjoint ( a::L1Hook )( θ ) = θ, Δ -> ( nothing, ΔL1( Δ, θ, a.λ ) )
Base.show(io::IO, l::L1Hook) = print(io, "L1 Penalty (λ = $(l.λ))")

struct L2Hook{F} <: GradientHook
    λ::F
end
ΔL2(Δ, θ, λ) = Δ .+ ( λ .* sum(abs2, θ ) ) 
( a::L2Hook )( x ) = x
@adjoint ( a::L2Hook )( θ ) = θ, Δ -> ( nothing, ΔL2( Δ, θ, a.λ ) )
Base.show(io::IO, l::L2Hook) = print(io, "L2 Penalty (λ = $(l.λ))")

struct ReverseGradientHook <: GradientHook; end
( a::ReverseGradientHook )( x ) = x
@adjoint ( a::ReverseGradientHook )( θ ) = θ, Δ -> (nothing, -Δ)
Base.show(io::IO, l::ReverseGradientHook) = print(io, "Reverse Gradient")

struct ClampHook{F} <: GradientHook
    ϵ::F
end 
( a::ClampHook )( x ) = x
@adjoint ( a::ClampHook )( θ ) = θ, Δ -> ( nothing, clamp.( Δ, -a.ϵ, a.ϵ ) )
Base.show(io::IO, a::ClampHook) = print(io, "Gradient Clamp (Δ ∈ [-$(a.ϵ),$(a.ϵ)])")
