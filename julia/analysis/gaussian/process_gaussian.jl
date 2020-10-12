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
      where(:n_iters .== 2000) |>
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
n_particles = 100
overwrite = true

res = @linq all_res |>
      where(:n_iters .== 2000) |>
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
    Plots.plot(title = "Time vs dims", xlabel = "Dim", ylabel = "Time [s]", legend = false, yaxis = :log)
p_μ = Plots.plot(
    title = "Mean error vs dims",
    xlabel = "Dim",
    ylabel = L"\|\mu -\mu_{true}\|",
    legend = false,
)
p_Σ = Plots.plot(
    title = "Cov error vs dims",
    xlabel = "Dim",
    ylabel = L"\|\Sigma -\Sigma_{true}\|",
    legend = false,
)
p_W = Plots.plot(
    title = "Wasserstein distance",
    xlabel = "Dim",
    ylabel = "W²",
    legend = false,
)
for (i, alg) in enumerate(algs)
    @info "Processing $(alg)"
    ft_alg = Symbol("ft_", alg)
    ft_var_alg = Symbol("ft_var_", alg)
    fm_alg = Symbol("fm_", alg)
    fm_var_alg = Symbol("fm_var_", alg)
    fv_alg = Symbol("fv_", alg)
    fv_var_alg = Symbol("fv_var_", alg)
    fw_alg = Symbol("fw_", alg)
    fw_var_alg = Symbol("fw_var_", alg)
    @eval begin
        global $(ft_alg) = []
        global $(ft_var_alg) = []
        global $(fm_alg) = []
        global $(fm_var_alg) = []
        global $(fv_alg) = []
        global $(fv_var_alg) = []
        global $(fw_alg) = []
        global $(fw_var_alg) = []
    end
    t_alg = Symbol("t_", alg)
    t_var_alg = Symbol("t_var_", alg)
    m_alg = Symbol("m_", alg)
    m_var_alg = Symbol("m_var_", alg)
    v_alg = Symbol("v_", alg)
    v_var_alg = Symbol("v_var_", alg)
    w_alg = Symbol("w_", alg)
    w_var_alg = Symbol("w_var_", alg)
    for j = 1:nrow(res)
        truth = res.d_target[j]
        @info "Row $j (dim = $(res.dim[j]))"
        @info res.path[j]
        @eval begin
            $(t_alg), $(t_var_alg) = process_time(res.$(alg)[$j])
            $(m_alg), $(m_var_alg) = process_means(res.$(alg)[$j], $(truth.m))
            $(v_alg), $(v_var_alg) =
                process_fullcovs(res.$(alg)[$j], vec($(truth.C.L) * $(truth.C.U)))
            # $(w_alg), $(w_var_alg) = process_wasserstein(res.$(alg)[$j], truth)
        end
        @eval begin
            push!($(ft_alg), last($(t_alg)))
            push!($(ft_var_alg), last($(t_var_alg)))
            push!($(fm_alg), last($(m_alg)))
            push!($(fm_var_alg), last($(m_var_alg)))
            push!($(fv_alg), last($(v_alg)))
            push!($(fv_var_alg), last($(v_var_alg)))
        end
    end
    #
    @eval Plots.plot!(
        p_t,
        dims[s],
        $(ft_alg)[s],
        #ribbon = sqrt.($(ft_var_alg)[s]),
        label = $(labels[alg]),
        color = $(dcolors[alg]),
    )
    @eval Plots.plot!(
        p_μ,
        dims[s],
        $(fm_alg)[s],
        ribbon = sqrt.($(fm_var_alg)[s]),
        label = $(labels[alg]),
        color = $(dcolors[alg]),
    )
    @eval Plots.plot!(
        p_Σ,
        dims[s],
        $(fv_alg)[s],
        ribbon = sqrt.($(fv_var_alg)[s]),
        label = $(labels[alg]),
        color = $(dcolors[alg]),
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
savefig(plotsdir("gaussian", "plots_vs_dim_" * plotname * ".png"))
display(p)
#
if overwrite
    cp(
        plotsdir("gaussian", "plots_vs_dim_" * plotname * ".png"),
        joinpath(
            "/home",
            "theo",
            "Tex Projects",
            "GaussianParticleFlow",
            "figures",
            "gaussian",
            "plots_vs_dim_" * plotname * ".png",
        ),
        force = true,
    )
end
