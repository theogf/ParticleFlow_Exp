using DrWatson
@quickactivate
include(projectdir("analysis", "post_process.jl"))


## Load data
all_res = collect_results(datadir("results", "gaussian"));
text_natmu = Dict(
        true => " - NG",
        false => "",
)

## Treat one convergence file
function plot_gaussian(
    n_dim, 
    cond,
    eta=1e-3;
    all_res = all_res,
    show_std_dev = false,
    show_lgd = true
)
    res = @linq all_res |>
        where(:n_dim .== n_dim) |>
        where(:eta .== eta) |>
        where(:n_iters .> 3000) |>
        where(:n_particles .== 0) |>
        where(:cond .== cond) |>
        where(:opt_stoch .=== Symbol(Descent))
    @info "Total of $(nrow(res)) for given parameters"
    if nrow(res) == 0
        @warn "Results for n_dim=$n_dim, cond=$cond not available yet"
        return nothing
    end
    d_res = Dict()
    # nrow(res) == 1 || error("Number of rows is not unique or is empty")
    for alg in algs
        # d_res[alg] = @linq res |> where(endswith.(:path, Regex("$(alg).*bson")))
        d_res[alg] = @linq res |> where(:alg .=== alg) # endswith.(:path, Regex("$(alg).*bson")))
    end
    d_vals = Dict()
    params_truth = BSON.load(datadir("exp_raw", "gaussian", savename(@dict(cond, n_dim), "bson")))
    truth = MvNormal(params_truth[:μ_target], params_truth[:Σ_target])
    # Plotting
    
    ylog = true
    ymin = eps(Float64)
    ymax = 1e4
    p_μ = Plots.plot(
        title = "Convergence Mean",
        xlabel = "Time [s]",
        ylabel = L"\|\mu - \mu_{true}\|",
        xaxis = :log,
        ylims = (ymin, ymax),
        yaxis = ylog ? (!show_std_dev ? :log : linear) : linear,
        legend = false,
    )
    p_Σ = Plots.plot(
        title = "Convergence Covariance",
        xlabel = "Time [s]",
        ylabel = L"\|\Sigma - \Sigma_{true}\|",
        xaxis = :log,
        ylims = (ymin, ymax),
        yaxis = ylog ? (!show_std_dev ? :log : linear) : linear,
        legend = false,
    )
    p_legend = Plots.plot(
        showaxis=false,
        hidedecorations=true,
        grid=false,
        legendfontsize=10.0
    )
    p_title = plot(title="D=$n_dim, κ=$(cond)", grid=false, showaxis=false)

    for (i, alg) in enumerate(algs)
        @info "Processing $(alg)"
        d = d_res[alg]
        for row in eachrow(d)
            vals = row.vals 
            try 
                process_means(vals, mean(truth))
            catch e
                @show vals
            end
            m, m_v = process_means(vals, mean(truth))
            C, C_v = process_fullcovs(vals, vec(cov(truth)))
            t, t_v = process_time(vals, Val(alg))
            Plots.plot!(
                p_μ,
                t,
                m,
                ribbon = show_std_dev ? sqrt.(m_v) : nothing,
                label = string(alg_lab[alg], text_natmu[row.natmu]),
                color = alg_col[alg],
                linestyle = alg_line[row.natmu],
            )
            Plots.plot!(
                p_Σ,
                t,
                C,
                ribbon = show_std_dev ? sqrt.(C_v) : nothing,
                label = string(alg_lab[alg], text_natmu[row.natmu]),
                color = alg_col[alg],
                linestyle = alg_line[row.natmu],
            )
            Plots.plot!(
                p_legend,
                [],
                [],
                label = string(alg_lab[alg], text_natmu[row.natmu]),
                color = alg_col[alg],
                linestyle = alg_line[row.natmu],
            )
        end
    end
    if show_lgd
        p = Plots.plot(p_title, p_legend, p_μ, p_Σ, layout=@layout([A{0.01h}; [B C D]]))
    else
        p = Plots.plot(p_title, p_μ, p_Σ, layout=@layout([A{0.01h}; [B C]]))
    end
    return p
end
mkpath(plotsdir("gaussian"))
for n_dim in 50, #[5,  10, 20, 50, 100], 
    cond in 1#[1, 10, 100]
    p = plot_gaussian(n_dim, cond, 0.001; show_std_dev=false, show_lgd=true)
    try
        display(p)
    catch e
        @warn "Plot was empty for n_dim=$n_dim and cond=$cond"
        continue
    end
    !isnothing(p) ? savefig(plotsdir("gaussian", savename(@dict(n_dim, cond), ".png"))) : nothing
end

