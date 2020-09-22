using DrWatson
@quickactivate
include(projectdir("analysis", "post_process.jl"))


## Load data
all_res = collect_results(datadir("results", "gaussian"))


## Treat one convergence file
res = @linq all_res |> where(:dim .== 5) |> where(:n_iters .== 5000) |> where(:n_particles .== 11) |> where(:n_runs .== 10)

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

## Treating all results at once

res = @linq all_res |> where(:n_iters .== 2000) |> where(:n_runs .== 10) |> where(:full_cov .== false)
dims = Float64.(Vector(res.dim))
s = sortperm(dims)
## Plot combined results
p_t = plot(title="Time vs dims", xlabel = "Dim", ylabel = "Time [s]")
p_μ = plot(title="Mean error vs dims", xlabel = "Dim", ylabel =L"\|\mu -\mu_{true}\|")
p_Σ = plot(title="Cov error vs dims", xlabel = "Dim", ylabel = L"\|\Sigma -\Sigma_{true}\|")
for (i, alg) in enumerate(algs)
    @info "Processing $(alg)"
    ft_alg = Symbol("ft_", alg); ft_var_alg = Symbol("ft_var_", alg)
    fm_alg = Symbol("fm_", alg); fm_var_alg = Symbol("fm_var_", alg)
    fv_alg = Symbol("fv_", alg); fv_var_alg = Symbol("fv_var_", alg)
    @eval begin
        global $(ft_alg) = []; global $(ft_var_alg) = [];
        global $(fm_alg) = []; global $(fm_var_alg) = [];
        global $(fv_alg) = []; global $(fv_var_alg) = [];
    end
    t_alg = Symbol("t_", alg); t_var_alg = Symbol("t_var_", alg)
    m_alg = Symbol("m_", alg); m_var_alg = Symbol("m_var_", alg)
    v_alg = Symbol("v_", alg); v_var_alg = Symbol("v_var_", alg)
    for j in 1:nrow(res)
        truth = res.d_target[j]
        @info "Row $j (dim = $(res.dim[j]))"
        @eval $(t_alg), $(t_var_alg) = process_time(res.$(alg)[$j])
        @eval $(m_alg), $(m_var_alg) = process_means(res.$(alg)[$j], $(truth.m))
        @eval $(v_alg), $(v_var_alg) = process_fullcovs(res.$(alg)[$j], vec($(truth.C.L) * $(truth.C.U)))
        @eval begin
            push!($(ft_alg), last($(t_alg)))
            push!($(ft_var_alg), last($(t_var_alg)))
            push!($(fm_alg), last($(m_alg)))
            push!($(fm_var_alg), last($(m_var_alg)))
            push!($(fv_alg), last($(v_alg)))
            push!($(fv_var_alg), last($(v_var_alg)))
        end
    end
    @eval plot!(p_t, dims[s], $(ft_alg)[s], ribbon = sqrt.($(ft_var_alg)[s]), label = $(labels[alg]))
    @eval plot!(p_μ, dims[s], $(fm_alg)[s], ribbon = sqrt.($(fm_var_alg)[s]), label = $(labels[alg]))
    @eval plot!(p_Σ, dims[s], $(fv_alg)[s], ribbon = sqrt.($(fv_var_alg)[s]), label = $(labels[alg]))
end
plot(p_t, p_μ, p_Σ, legend = false)
