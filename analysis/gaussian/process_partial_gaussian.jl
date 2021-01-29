using DrWatson
@quickactivate
include(projectdir("analysis", "post_process.jl"))


## Load data
all_res = collect_results(datadir("results", "gaussian", "partial"))

## Treat one convergence file
# function plot_partial_gaussian(n_dim = 50, cond = 1; all_res = all_res)
function create_partial_dict(n_dim = 100, cond = 1)
    res = @linq all_res |>
        where(:n_dim .== n_dim) |>
        #   where(:n_iters .== 3000) |>
        where(:cond .== cond)
    @info "Total of $(nrow(res)) for given parameters"
    if nrow(res) == 0
        @warn "Results for n_dim=$n_dim, cond=$cond, not available yet"
        return nothing
    end
    d_res = Dict()
    d_truth = BSON.load(datadir("exp_raw", "gaussian", savename(@dict(cond, n_dim), "bson")))
    truth = MvNormal(d_truth[:μ_target], d_truth[:Σ_target])
    d_res[:n_particles] = zeros(nrow(res))
    d_res[:Δm] = zeros(nrow(res))
    d_res[:varm] = zeros(nrow(res))
    d_res[:ΔC] = zeros(nrow(res))
    d_res[:varC] = zeros(nrow(res))
    d_res[:Δ] = zeros(nrow(res))
    d_res[:varΔ] = zeros(nrow(res))
    d_res[:t] = zeros(nrow(res))
    d_res[:t_v] = zeros(nrow(res))
    for (i, row) in enumerate(eachrow(res))
        vals = row.vals
        d_res[:n_particles][i] = row.n_particles
        d_res[:t][i], d_res[:t_v][i] = first.(process_time(vals))
        d_res[:Δm][i], d_res[:varm][i], d_res[:ΔC][i], d_res[:varC][i], d_res[:Δ][i], d_res[:varΔ][i] = process_means_plus_covs(vals, truth)
    end

    return d_res
end


d = Dict()
for n_dim in [100, 500, 1000], cond in [1, 5, 10]
    d[(n_dim, cond)] = create_partial_dict(n_dim, cond)
    # !isnothing(p) ? savefig(plotsdir("gaussian", savename(@dict(n_dim, cond), ".png"))) : nothing
end


## Do the plots

function add_plots!(p_m, p_C, p_Δ, p_t, d, cond, n_dim; show_std_dev=true)
    if isnothing(d)
        @warn "Empty dict for cond=$cond and n_dim=$n_dim"
    end
    order = sortperm(d[:n_particles])
    plot!(
        p_m,
        d[:n_particles][order],
        d[:Δm][order],
        ribbon = show_std_dev ? d[:varm] : nothing,
        label = "κ = $cond"
    )
    plot!(
        p_C,
        d[:n_particles][order],
        d[:ΔC][order],
        ribbon = show_std_dev ? d[:varC] : nothing,
        label = "κ = $cond"
    )
    plot!(
        p_Δ,
        d[:n_particles][order],
        d[:Δ][order],
        ribbon = show_std_dev ? d[:varΔ] : nothing,
        label = "κ = $cond"
    )
    plot!(
        p_t,
        d[:n_particles][order],
        d[:t][order],
        ribbon = show_std_dev ? d[:t_v] : nothing,
        label = "κ = $cond"
    )
end




show_std_dev = true
show_lgd = false
ylog = true
ymin = eps(Float64)
ymax = 1e4
n_dim = 100
p_m = Plots.plot(
    title = "Convergence Mean",
    xlabel = "# Particles",
    ylabel = L"\|m - μ\|",
    xaxis = :log,
    # ylims = (ymin, ymax),
    # yaxis = ylog ? (!show_std_dev ? :log : linear) : linear,
    legend = show_lgd,
)
p_C = Plots.plot(
    title = "Convergence Covariance",
    xlabel = "# Particles",
    ylabel = L"\|C - \Sigma\|",
    xaxis = :log,
    # ylims = (ymin, ymax),
    # yaxis = ylog ? (!show_std_dev ? :log : linear) : linear,
    legend = show_lgd,
)
p_Δ = Plots.plot(
    title = "Convergence",
    xlabel = "# Particles",
    ylabel = L"\|m -\mu \| + \|C - \Sigma\|",
    xaxis = :log,
    # ylims = (ymin, ymax),
    # yaxis = ylog ? (!show_std_dev ? :log : linear) : linear,
    legend = show_lgd,
)
p_t = Plots.plot(
    title = "Algorithm cost",
    xlabel = "# Particles",
    ylabel = "Time [s]",
    xaxis = :log,
    # ylims = (ymin, ymax),
    # yaxis = ylog ? (!show_std_dev ? :log : linear) : linear,
    legend = show_lgd,
)
p_title = plot(title="D=$n_dim", grid=false, showaxis=false)
for cond in [1, 5, 10]
    add_plots!(p_m, p_C, p_Δ, p_t, d[(n_dim, cond)], cond, n_dim, show_std_dev=show_std_dev)
end

p = Plots.plot(p_title, p_m, p_C, p_Δ, p_t, layout=@layout([A{0.01h}; [B C; E F]]))
display(p)
mkpath(plotsdir("partial"))
savefig(plotsdir("partial", "D=$(n_dim).png"))