using DrWatson
@quickactivate
include(projectdir("analysis", "post_process.jl"))


## Load data
all_res = collect_results(datadir("results", "gaussian"))

## Treat one convergence file
gdim = 80;
n_p = 0;
full_cov = true
res = @linq all_res |>
      where(:dim .== gdim) |>
      where(:n_iters .== 3000) |>
      where(:n_particles .== (iszero(n_p) ? gdim + 1 : n_p)) |>
      where(:full_cov .== full_cov)
@assert nrow(res) == 1 "Number of rows is not unique or is empty"

truth = first(res.d_target)
# Plotting

p_μ = Plots.plot(
    title = "Convergence Mean",
    xlabel = "Time [s]",
    ylabel = L"\|\mu - \mu_{true}\|",
    xaxis = :log,
)
p_Σ = Plots.plot(
    title = "Convergence Covariance",
    xlabel = "Time [s]",
    ylabel = L"\|\Sigma - \Sigma_{true}\|",
    xaxis = :log,
)
for (i, alg) in enumerate(algs)
    @info "Processing $(alg)"
    t_alg = Symbol("t_", alg)
    t_var_alg = Symbol("t_var_", alg)
    m_alg = Symbol("m_", alg)
    m_var_alg = Symbol("m_var_", alg)
    v_alg = Symbol("v_", alg)
    v_var_alg = Symbol("v_var_", alg)
    @eval $(t_alg), $(t_var_alg) = process_time(first(res.$(alg)))
    @eval $(m_alg), $(m_var_alg) = process_means(first(res.$(alg)), truth.m)
    @eval $(v_alg), $(v_var_alg) =
        process_fullcovs(first(res.$(alg)), vec(truth.C.L * truth.C.U))
    @eval Plots.plot!(
        p_μ,
        $(t_alg),
        $(m_alg),
        ribbon = sqrt.($(m_var_alg)),
        label = $(labels[alg]),
        color = colors[$i],
    )
    @eval Plots.plot!(
        p_Σ,
        $(t_alg),
        $(v_alg),
        ribbon = sqrt.($(v_var_alg)),
        label = $(labels[alg]),
        color = colors[$i],
    )
end
display(Plots.plot(p_μ, p_Σ, legend = false))

## Treating all dimensions at once

fullcov = false
n_particles = 0
overwrite = true

res = @linq all_res |>
      where(:n_iters .== 3000) |>
      where(:n_runs .== 10) |>
      where(:full_cov .== fullcov)


res = if n_particles == 0
    @linq res |> where(:n_particles .== :dim .+ 1)
else
    @linq res |> where(:n_particles .== n_particles)
end
dims = Float64.(Vector(res.dim))
s = sortperm(dims)
# Plot combined results
p_t =
    Plots.plot(title = "",#Time vs dims",
    xlabel = "D", ylabel = "Time [s]", legend = false, yaxis = :log)
p_μ = Plots.plot(
    title = "",#Mean error vs dims",
    xlabel = "D",
    ylabel = L"\|\mu -\mu_{true}\|²",
    legend = false,
)
p_Σ = Plots.plot(
    title = "",#Cov error vs dims",
    xlabel = "D",
    ylabel = L"\|\Sigma -\Sigma_{true}\|²",
    legend = false,
)
p_W = Plots.plot(
    title = "",#Wasserstein distance",
    xlabel = "D",
    ylabel = "W₂",
    legend = false,
)
ft = Dict(:mean=>Dict(), :var=>Dict())
fm = Dict(:mean=>Dict(), :var=>Dict())
fw = Dict(:mean=>Dict(), :var=>Dict())
for (i, alg) in enumerate(algs)
    @info "Processing $(alg)"
    ft[:mean][alg] = []
    ft[:var][alg] = []
    fm[:mean][alg] = []
    fm[:var][alg] = []
    fw[:mean][alg] = []
    fw[:var][alg] = []
    for j = 1:nrow(res)
        truth = res.d_target[j]
        @info "Row $j (dim = $(res.dim[j]))"
        @info res.path[j]
        t_alg, t_var_alg = process_time(res.$(alg)[j])
        m_alg, m_var_alg = process_means(res.$(alg)[j], truth.m)
        v_alg, v_var_alg = process_fullcovs(res.$(alg)[j], vec(truth.C.L * truth.C.U))
        push!(ft[:mean][alg], last(t_alg))
        push!(ft[:var][alg], last(t_var_alg))
        push!(fm[:mean][alg], last(m_alg))
        push!(fm[:var][alg], last(m_var_alg))
        push!(fv[:mean][alg], last(v_alg))
        push!(fv[:var][alg], last(v_var_alg))
    end
    #
    Plots.plot!(
        p_t,
        dims[s],
        ft[:mean][alg][s],
        #ribbon = sqrt.($(ft_var_alg)[s]),# removed because of logscale
        label = labels[alg],
        color = dcolors[alg],
    )
    Plots.plot!(
        p_μ,
        dims[s],
        fm[:mean][alg][s],
        ribbon = sqrt.(fm[:var][alg][s]),
        label = labels[alg],
        color = dcolors[alg],
    )
    Plots.plot!(
        p_Σ,
        dims[s],
        fv[:mean][alg][s],
        ribbon = sqrt.(fv[:var][alg][s]),
        label = labels[alg],
        color = dcolors[alg],
    )
end
pleg = Plots.plot(
    [[], [], []],
    [[], [], []],
    ribbon = [],
    label = reshape(getindex.(Ref(labels), algs), 1, :),
    color = reshape(getindex.(Ref(dcolors), algs), 1, :),
    framestyle = :none,
    legend = :top,
)
p = Plots.plot(p_t, p_μ, p_Σ, pleg)
plotname = @savename fullcov n_particles
savepath = plotsdir("gaussian")
mkpath(savepath)
savefig(joinpath(savepath, "plots_vs_dim_" * plotname * ".png"))
display(p)
