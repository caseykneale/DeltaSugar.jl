#:D why not....
nth = [:second, :third, :fourth, :fifth, :sixth, :seventh, :eigth, :ninth, :tenth, 
    :eleventh, :twelfth, :thirteenth, :fourteenth, :fifteenth, :sixteenth, :seventeenth,
    :eighteenth, :nineteenth, :twentieth]

for (i,n) in enumerate(nth)
    @eval $n(x) = x[$(i+1)]
end