using DeltaSugar, Test, Flux

function reverse_grad_tester(d0,d1,d2)
    return all(-(d0.weight - d1.weight) .≈ (d0.weight - d2.weight)) &&
        all(-(d0.bias - d1.bias) .≈ (d0.bias - d2.bias)) 
end

loss( x1, x2, m ) = sum( abs2, ( m( x1 ) - x2 ) .^ 2 )
mloss( x1, x2, m ) = sum( abs2, ( m( x1 ) - x2 ) .^ 2 ) / length(x2)

@testset "TenseLayer" begin
    a = randn( 10, 6 )
    b = randn( 10, 3 )
    model = TenseLayer( 6, 3 )
    @test all(size( model( a ) ) .== ( 10, 3 ) )

    #ensure the gradient moves
    w0 = deepcopy(model.weight)
    train!(a, b, model, Flux.params(model), loss, Descent(0.01))
    @test sum(abs, w0 - model.weight) > 0.0
end

@testset "TenseLayer - Reverse Gradient Check" begin
    a,b = randn( 10, 6 ), randn( 10, 3 )
    original      = TenseLayer( 6, 3 )
    model_forward = deepcopy(original)
    model_reverse = RevGrad( deepcopy(original) )
    @test all(size( model_forward( a ) ) .== ( 10, 3 ) )
    @test all(size( model_reverse( a ) ) .== ( 10, 3 ) )
    #ensure the gradient moves
    ps = Flux.params( model_forward )
    train!(a,b, model_forward, ps, loss, Descent(1.))
    ps = Flux.params( model_reverse )
    train!(a,b, model_reverse, ps, loss, Descent(1.))
    @test reverse_grad_tester(original, model_forward, model_reverse)
    #now a 2 layer model check...
    a, b = randn( 10, 6 ), randn( 10 )
    layer1, layer2 = TenseLayer( 6, 3 ), TenseLayer( 3, 1 ) 
    model_forward = Chain( deepcopy(layer1), deepcopy(layer2) )
    model_reverse = Chain( RevGrad(layer1), RevGrad(layer2) )
    #ensure the gradient moves
    ps = Flux.params( model_forward )
    train!(a,b, model_forward, ps, loss, Descent(1.))
    ps = Flux.params( model_reverse )
    train!(a,b, model_reverse, ps, loss, Descent(1.))
    @test reverse_grad_tester(layer1, model_forward[1], model_reverse[1])
    @test reverse_grad_tester(layer2, model_forward[2], model_reverse[2])
    
    #test that the chain operator works
    layer1, layer2 = TenseLayer( 6, 3 ), TenseLayer( 3, 1 ) 
    model_forward = Chain( deepcopy(layer1), deepcopy(layer2) )
    model_reverse = RevGrad( model_forward )
    ps = Flux.params( model_forward )
    train!(a,b, model_forward, ps, loss, Descent(1.))
    ps = Flux.params( model_reverse )
    train!(a,b, model_reverse, ps, loss, Descent(1.))
    @test reverse_grad_tester(layer1, model_forward[1], model_reverse[1])
    @test reverse_grad_tester(layer2, model_forward[2], model_reverse[2])
end

@testset "DenseLayer" begin
    a = randn( 6, 10 )
    b = randn( 3, 10 )
    model = DenseLayer( 6, 3 )
    @test all(size( model( a ) ) .== ( 3, 10 ) )
    #ensure the gradient moves
    w0 = deepcopy(model.weight)
    train!(a, b, model, Flux.params(model), loss, Descent(0.01))
    @test sum(abs, w0 - model.weight) > 0.0
end

@testset "DenseLayer - Reverse Gradient Check" begin
    a = randn( 6, 10 )
    b = randn( 3, 10 )
    original      = DenseLayer( 6, 3 )
    model_forward = deepcopy(original)
    model_reverse = RevGrad( deepcopy(original) )
    @test all(size( model_forward( a ) ) .== ( 3, 10 ) )
    @test all(size( model_reverse( a ) ) .== ( 3, 10 ) )
    #ensure the gradient moves
    ps = Flux.params( model_forward )
    train!(a,b, model_forward, ps, loss, Descent(1.))
    ps = Flux.params( model_reverse )
    train!(a,b, model_reverse, ps, loss, Descent(1.))
    @test reverse_grad_tester(original,model_forward,model_reverse)

    #now a 2 layer model check...
    a, b = randn( 6, 10 ), randn( 1, 10 )
    layer1, layer2 = DenseLayer( 6, 3 ), DenseLayer( 3, 1 ) 
    model_forward = Chain( deepcopy(layer1), deepcopy(layer2) )
    model_reverse = Chain( RevGrad(layer1), RevGrad(layer2) )
    #ensure the gradient moves
    ps = Flux.params( model_forward )
    train!(a,b, model_forward, ps, loss, Descent(1.))
    ps = Flux.params( model_reverse )
    train!(a,b, model_reverse, ps, loss, Descent(1.))
    @test reverse_grad_tester(layer1, model_forward[1], model_reverse[1])
    @test reverse_grad_tester(layer2, model_forward[2], model_reverse[2])
    
    #test that the chain operator works
    layer1, layer2 = DenseLayer( 6, 3 ), DenseLayer( 3, 1 ) 
    model_forward = Chain( deepcopy(layer1), deepcopy(layer2) )
    model_reverse = RevGrad( model_forward )
    ps = Flux.params( model_forward )
    train!(a,b, model_forward, ps, loss, Descent(1.))
    ps = Flux.params( model_reverse )
    train!(a,b, model_reverse, ps, loss, Descent(1.))
    @test reverse_grad_tester(layer1, model_forward[1], model_reverse[1])
    @test reverse_grad_tester(layer2, model_forward[2], model_reverse[2])
end

@testset "Gradient Clip Test" begin
    a = randn( 6, 10 )
    b = randn( 3, 10 )
    ϵ = 1e-2
    model = GradientClamp( DenseLayer( 6, 3 ), ϵ )
    @test all(size( model( a ) ) .== ( 3, 10 ) )
    #ensure the gradient is clipped
    ps = Flux.Params(Flux.params(model))
    gs = Flux.gradient(ps) do
        loss(a, b, model)
    end
    @test all( [ reduce(max, g) <= ϵ for (_,g) in gs.grads ])
    @test all( [ reduce(min, g) >= -ϵ for (_,g) in gs.grads ])

    ϵ = 100.0
    model = GradientClamp( DenseLayer( 6, 3 ), ϵ )
    @test all(size( model( a ) ) .== ( 3, 10 ) )
    #ensure the gradient is clipped
    ps = Flux.Params(Flux.params(model))
    gs = Flux.gradient(ps) do
        loss(a, b, model)
    end
    #println.([ g for (_,g) in gs.grads ])
    @test all( [ reduce(max, g) <= ϵ for (_,g) in gs.grads ])
    @test all( [ reduce(min, g) >= -ϵ for (_,g) in gs.grads ])
end

@testset "Ridge Test" begin
    a = randn( 10, 20 )
    b = randn( 1, 20 )
    a[5,:]  = 0.5 * b[:]
    model = L2( DenseLayer( 10, 1 ), 0.02 )
    weight_i = deepcopy( model.weight )
    @test all(size( model( a ) ) .== ( 1, 20 ) )
    ps = Flux.params( model )
    for _ in 1:100
        train!( a, b, model, ps, mloss, Descent( 0.01 ) )
    end
    @test sum(abs2, model.weight ) < sum(abs2, weight_i )
end

@testset "FF Chain Sugar Tests" begin
    z = PlainChain( 5,4,3,2,1 ; σ = identity )
    @test all(size(z(randn(5,10))) .== (1,10))

    x = TChain( 5,4,3,2,1 ; σ = identity )
    @test all(size(x(randn(10,5))) .== (10,1))
end

@testset "LASSO Test" begin
    """
    This one won't always work, c'est la vie...
    """
    a = randn( 10, 20 )
    b = randn( 1, 20 )
    a[5,:]  = 0.5 * b[:]
    model = L1( DenseLayer( 10, 1 ), 0.00001 )
    #model.weight[1,5] .+= 0.1
    @test all(size( model( a ) ) .== ( 1, 20 ) )
    ps = Flux.params( model )
    for _ in 1:10000
        train!( a, b, model, ps, mloss, Descent( 0.001 ) )
    end
    #println( model.weight )
    @test argmax( model.weight[:] ) == 5
end
