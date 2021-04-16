macro memoize(cache_var)
    filename = String(cache_var) * ".jld2"
    quote
        if !isfile( $filename )
            @save $filename $cache_var
        else 
            @load $filename $cache_var
        end
    end
end

macro memoize(cache_var, fn)
    filename = String(cache_var) * ".jld2"
    quote    
        if !isfile( $filename )
            $fn
            @save $filename $cache_var
        else
            @load $filename $cache_var
        end
    end
end