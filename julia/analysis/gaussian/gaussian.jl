using DrWatson
@quickactivate
include(projectdir("analysis", "post_process.jl"))


## Load data and filter it
all_res = collect_results(datadir("results", "gaussian"))

res = @linq all_res |> where(:dim .== 10) |> where(:n_iters .== 1000) |> where(:n_particles .== 11) |> where(:n_runs .== 10)

@assert nrow(res) == 1 "Number of rows is not unique or is empty"

truth = first(res.d_target)
## Plotting

p_μ = plot(title = "Convergence Mean", xlabel = "Time [s]", ylabel =L"\|\mu - \mu_{true}\|", xaxis=:log)
p_Σ = plot(title = "Convergence Covariance", xlabel = "Time [s]", ylabel =L"\|\Sigma - \Sigma_{true}\|", xaxis=:log)
for (i, alg) in enumerate(algs)
    @info "Processing $(alg)"
    t_alg = Symbol("t_", alg); t_var_alg = Symbol("t_var_", alg)
    m_alg = Symbol("m_", alg); m_var_alg = Symbol("m_var_", alg)
    v_alg = Symbol("v_", alg); v_var_alg = Symbol("v_var_", alg)
    @eval $(t_alg), $(t_var_alg) = process_time(first(res.$(alg)))
    @eval $(m_alg), $(m_var_alg) = process_means(first(res.$(alg)), truth.m)
    @eval $(v_alg), $(v_var_alg) = process_fullcovs(first(res.$(alg)), vec(truth.C.L * truth.C.U))
    @eval plot!(p_μ, $(t_alg), $(m_alg), ribbon = sqrt.($(m_var_alg)), label = $(labels[alg]), color = colors[$i])
    @eval plot!(p_Σ, $(t_alg), $(v_alg), ribbon = sqrt.($(v_var_alg)), label = $(labels[alg]), color = colors[$i])
end
display(plot(p_μ, p_Σ))
