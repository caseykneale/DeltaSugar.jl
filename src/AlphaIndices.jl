#:D why not....
nth = ["second","third","fourth","fifth","sixth","seventh","eigth","ninth","tenth",
        "eleventh","twelfth","thirteenth","fourteenth","fifteenth","sixteenth","seventeenth",
        "eighteenth", "nineteenth", "twentieth"] .|> Symbol

for (i,n) in enumerate(nth)
    idx = i+1
    @eval $n(x) = x[$idx]
end