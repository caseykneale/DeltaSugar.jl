module DeltaSugar
    using Base, Flux
    using Zygote: @adjoint

    abstract type SugarLayer; end
    abstract type GradientHook; end

    include("GradientModifiers.jl")
    export composer
    export GradientHook, IdentityHook, 
            L1Hook, L2Hook, 
            ReverseGradientHook, GradientClamp

    include.( [ "DenseLayer.jl", "TenseLayer.jl" ] )
    export DenseLayer, show, 
            TenseLayer#, show

    include("LayerSugar.jl")
    export L1, L2, RevGrad, ClampHook,
            L1!, L2!, RevGrad!, ClampHook!

    include("SweeterChains.jl")
    export PlainChain, TChain

    include("Train.jl")
    export train_light!, smart_yield

    #debauchery
    include("BaseOperators.jl")
    export lessapprox, gtapprox, ≲, ⪅, ≳, ⪆, ⊥

    include("Memoize.jl")
    export memoize

    include("AlphaIndices.jl")
    export second,third,fourth,fifth,sixth,seventh,eigth,
            ninth,tenth,eleventh,twelfth,thirteenth,
            fourteenth,fifteenth,sixteenth,seventeenth,
            eighteenth,nineteenth,twentieth
end
