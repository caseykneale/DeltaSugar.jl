using DeltaSugar
using Documenter

DocMeta.setdocmeta!(DeltaSugar, :DocTestSetup, :(using DeltaSugar); recursive=true)

makedocs(;
    modules=[DeltaSugar],
    authors="Casey Kneale",
    repo="https://github.com/caseykneale/DeltaSugar.jl/blob/{commit}{path}#{line}",
    sitename="DeltaSugar.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://caseykneale.github.io/DeltaSugar.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/caseykneale/DeltaSugar.jl",
)
