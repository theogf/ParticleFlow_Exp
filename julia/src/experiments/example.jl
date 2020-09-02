using DrWatson
@quickactivate

include("./base.jl")

struct ExampleExperiment
    studyname::String # TODO remove any time
    base
end

# function experimentid(e::ExampleExperiment)
#     return string(e.base)
# end

function results(exp::ExampleExperiment)

    res = Dict()
    for exponent in [2, 3, 4]
        res["base$(exp.base)/expon$exponent"] = Dict(
            "base" => exp.base,
            "exponent" => exponent,
            "result" => exp.base^exponent
        )
    end

    return res
end

experiments = [ExampleExperiment("", b) for b in [4, 10, 100]]
runexperiment("example_study", experiments)

# example loading data
example_data = loaddata("example_study", 19:20)
