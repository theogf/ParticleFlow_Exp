using DrWatson
@quickactivate
include(projectdir("analysis", "post_process.jl"))


## Load data and filter it
all_res = collect_results(datadir("results", "gaussian"))

res = @linq all_res |> where(:dim .== 10) |> where(:n_iters .== 1000) |> where(:n_particles .== 11) |> where(:n_runs .== 10)

@assert nrow(res) == 1 "Number of rows is not unique or is empty"

truth = first(res.d_target)
## Plotting
algs = [:gpf, :advi, :steinvi]

p = plot(title = "Convergence Mean", xlabel = "Time [s]", ylabel =L"\mu - \mu_{true}", xaxis=:log)
for alg in algs
    @info "Processing $(alg)"
    t_alg = Symbol("t_", alg); t_var_alg = Symbol("t_var_", alg)
    m_alg = Symbol("m_", alg); m_var_alg = Symbol("m_var_", alg)
    @eval $(t_alg), $(t_var_alg) = process_time(first(res.$(alg)))
    @eval $(m_alg), $(m_var_alg) = process_means(first(res.$(alg)), truth)
    @eval plot!($(t_alg), $(m_alg), ribbon = sqrt.($(m_var_alg)), label = $(string(alg)))
end
display(p)
