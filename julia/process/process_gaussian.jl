using DrWatson; @quickactivate
include(projectdir("process", "post_process.jl"))

res = collect_results(datadir("results", "gaussian"))
