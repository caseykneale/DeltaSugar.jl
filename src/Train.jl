function train_light!(x, y, model, ps, loss, opt)
    ps = Flux.Params(ps)
    gs = Flux.gradient(ps) do
        loss(x, y, model)
    end
    Flux.update!(opt, ps, gs)
end

scalar_yield(iter::Int; every = 100) = (iter % every == 0) && yield()

function log2_yield(iter::Int; every = 100) 
    l2 = log2(iter)
    if (iter >= 0) && (trunc(l2) ≈ l2)
        yield()
    end
end

function smart_yield(iter::Int; every = 100)
    if (iter % every == 0) 
        yield()
    else
        l2 = log2(iter)
        if (iter > 0) && (trunc(l2) ≈ l2)
            yield()
        end
    end
end

function train_loop!(   X, Y, model, loss, opt; 
                        max_iters = 1000, yeild_fn = identity, record_every = 10)
    loss_record = []
    ps = Flux.params(model)
    for (iter,x,y) in zip(1:max_iters,X,Y)
        ps = Flux.Params(ps)
        gs = Flux.gradient(ps) do
            loss(x, y, model)
        end
        Flux.update!(opt, ps, gs)
        #for reporting out
        if iter % record_every == 0
            push!(loss_record)
        end
        #allow Ctrl+C interrupts after N iterations
        yeild_fn(iter)
    end
    return loss_record
end

#Do early stopping...