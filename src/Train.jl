function train_light!(x, y, model, ps, loss, opt)
    ps = Flux.Params(ps)
    gs = Flux.gradient(ps) do
        loss(x, y, model)
    end
    Flux.update!(opt, ps, gs)
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
