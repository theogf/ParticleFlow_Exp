using DrWatson
using JLD2
using Printf
using DataFrames

abstract type ExperimentSpecification end

# requires results(exp)
function runexperiment(studyname::String, experiments::Array)
    runid = _getnextrun(datadir(studyname))
    mkpath(datadir(studyname, runid))

    dir = datadir(studyname, runid, "gitcommit")
    println("Saving to $dir")
    open(dir, "w") do io
        println(io, gitdescribe())
    end

    dir = datadir(studyname, runid, "gitpatch")
    println("Saving to $dir")
    open(dir, "w") do io
        println(io, DrWatson.gitpatch())
    end

    dir = datadir(studyname, runid, "script")
    println("Saving to $dir")
    open(dir, "w") do io
        println(io, _getscript())
    end

    for exp in experiments
        res = results(exp)
        for (path, result) in res
            @assert ! (path in ["gitcommit", "gitpatch", "script"])

            dict = Dict()  # uptyped dictionary
            dict["__studyname"] = studyname
            dict["__runid"] = runid
            dict["__path"] = path

            for (k, v) in result
                @assert !haskey(dict, k)
                dict[k] = v
            end
            filename = "$(datadir(studyname, runid, path)).jld2"
            mkpath(dirname(filename))
            println("Saving to $filename")
            save(filename, dict)
        end
    end
end

function _getscript()
    "Returns the stacktrace from the call to runexperiments
    (6th element) to the top level scope"
    res = ""
    for stackframe in stacktrace()[6:end]
        if stackframe.func == Symbol("top-level scope")
            # don't need to go beyond the top-level scope
            return res
        end
    end
    @assert length(res) > 0
    res
end

function loaddata(studyname::String, runrange=:, filter=Nothing)
    runs = _runs(datadir(studyname))[runrange]
    dfs = []
    for run in runs
        folder = datadir(studyname, run)
        println("loading  $folder")
        df = collect_results(folder, subfolders=true)
        #println(df)
        push!(dfs, df)
    end
    vcat(dfs...) 
end


MAXRUN = 999
function _iterruns()
    i = 1
    Channel() do channel
        while i <= MAXRUN
            run_name = @sprintf "run%03d" i
            put!(channel, run_name)
            i += 1
        end
    end
end
    

function _runs(folder)
    res = []
    for runname in _iterruns()
        if ispath("$folder/$runname")
            push!(res, runname)
        else
            return res
        end
    end
end


function _getnextrun(folder)
    "Returns first run that has not yet been used"
    for runname in _iterruns()
        if !ispath("$folder/$runname")
            return runname
        end
    end
    throw("Maximum number of runs reached, remove old runs or create new study")
end


# function _getnextrun(folder)
#     num = 0
#     run_name = @sprintf RUNFORMAT num
#     while ispath("$folder/$run_name")
#         num += 1
#         run_name = @sprintf RUNFORMAT num
#     end
#     return  run_name
# end



"
Features:
- (done) store arbitrary dicts without overriding old data
- (TODO) able to recreate exact code and script that generated specific data
- (done) load data into data frame
- (TODO) flexible filtering when obtaining results

One run per script

The data format is the following, for each experiment.
Not 100% sure about this being the best possible format.

results_folder / study_name / run001 / exp0001 / it00010 <- results at specific iteration
                                               / it00040
                                               / it00020
                                     / gitcommit
                                     / gitpatch
                                     / script


Ways I could do it in python

with experiment_saver() as es:
    for options in myoptions():
        for i, result in results(options):
            es.save(gen_path(options, i), results(options))
"


##
#
# What are the use cases that we want su support?
# - x models, can be identified by just the model name in theory and logπ
# - different methods, in first instance can be \
#
# What should be stored in the above?
# - a plot could be a gif or similar
# - some performance metrics that can be measured across vi algos (but not necessarily across)
#
# Possible next extensions
# - what if we need in between
#
