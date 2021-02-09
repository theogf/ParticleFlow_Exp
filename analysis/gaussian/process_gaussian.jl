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
    show_lgd = true,
    use_quantile = true,
)
    res = @linq all_res |>
        where(:n_dim .== n_dim) |>
        where(:eta .== eta) |>
        where(:n_iters .> 20000) |>
        where(:n_particles .== 0) |>
        where(:cond .== cond) |>
        where(:opt_stoch .=== Symbol(Descent)) |> 
        where(:n_runs .== 10)
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
    
    ylog = :log
    ymin = eps(Float64)
    ymax = 1e2
    tfsize = 21.0
    p_μ = Plots.plot(
        title = cond == 1 ? L"\|m^t - \mu\|" : "",
        titlefontsize = tfsize,
        xlabel = cond == 100 ? "Time [s]" : "",
        ylabel = "",
        xaxis = :log,
        ylims = (ymin, ymax),
        yaxis = ylog,# ? (!show_std_dev ? :log : :linear) : :linear,
        legend = false,
    )
    annotate!(p_μ, 5e-2, 1e-10, Plots.text(latexstring("\\kappa = $cond"), :left, 18))
    p_Σ = Plots.plot(
        title = cond == 1 ? L"\|C^t- \Sigma\|" : "",
        titlefontsize = tfsize,
        xlabel = cond == 100 ? "Time [s]" : "",
        ylabel = "",
        xaxis = :log,
        ylims = (ymin, ymax),
        yaxis = ylog,# ? (!show_std_dev ? :log : :linear) : :linear,
        legend = false,
    )
    p_legend = Plots.plot(
        showaxis=false,
        hidedecorations=true,
        grid=false,
        legendfontsize=10.0
    )
    p_title = plot(title="D=$n_dim, κ=$(cond)", grid=false, showaxis=false)

    for (i, alg) in enumerate(alg_line_order)
        @info "Processing $(alg)"
        d = d_res[alg]
        for row in eachrow(d)
            vals = row.vals 
            if alg == :gf && row.natmu == true
                continue
            elseif alg == :svgd && row.opt_det == :RMSProp
                continue
            elseif alg == :iblr && row.comp_hess == :rep
                continue
            end
            m, m_v = process_means(vals, mean(truth), use_quantile=use_quantile)
            C, C_v = process_fullcovs(vals, vec(cov(truth)), use_quantile=use_quantile)
            t, t_v = process_time(vals, Val(alg))
            if use_quantile
                Plots.plot!(
                    p_μ,
                    t,
                    m,
                    fillrange= show_std_dev ? m_v : nothing,
                    fillalpha=0.3,
                    label=string(alg_lab[alg], text_natmu[row.natmu]),
                    color=alg_col[alg],
                    linestyle=alg_line[row.natmu]
                )
                Plots.plot!(
                    p_Σ,
                    t,
                    C,
                    fillrange = show_std_dev ? C_v : nothing,
                    fillalpha=0.3,
                    label = string(alg_lab[alg], text_natmu[row.natmu]),
                    color = alg_col[alg],
                    linestyle = alg_line[row.natmu],
                )
            else
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
            end
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
    display(p)
    return p, p_μ, p_Σ
end
mkpath(plotsdir("gaussian"))
ps = Dict()
D = 50
for n_dim in D, #[5,  10, 20, 50, 100], 
    cond in [1, 10, 100]
    ps[cond] = Dict()
    p, ps[cond][:μ], ps[cond][:Σ] = plot_gaussian(n_dim, cond, 0.01; show_std_dev=true, show_lgd=false, use_quantile=true)
    try
        display(p)
    catch e
        @warn "Plot was empty for n_dim=$n_dim and cond=$cond"
        continue
    end
    !isnothing(p) ? savefig(plotsdir("gaussian", savename(@dict(n_dim, cond), ".png"))) : nothing
end
## Working with the plots 
lloc = (0.1, 0.0)
lfsize = 16.0
p_legend1 = Plots.plot(
        showaxis=false,
        legend=lloc,
        hidedecorations=true,
        grid=false,
        legendfontsize=lfsize,
        fg_legend=:white,
        bg_legend=:white,
    )
for alg in algs[1:3]
    plot!(
        p_legend1,
        [],
        [],
        color=alg_col[alg],
        label=alg_lab[alg],
    )
end
p_legend1

p_legend2 = Plots.plot(
        legend=lloc,
        showaxis=false,
        hidedecorations=true,
        grid=false,
        legendfontsize=lfsize,
        fg_legend=:white,
        bg_legend=:white,
    )
for alg in algs[4:end]
    plot!(
        p_legend2,
        [],
        [],
        color=alg_col[alg],
        label=alg_lab[alg],
    )
end


p = plot(
    ps[1][:μ], ps[1][:Σ], ps[10][:μ], ps[10][:Σ], ps[100][:μ], ps[100][:Σ], p_legend1, p_legend2;
    dpi = 300,
    layout = @layout([A B;C D;E F;G{0.2h} H]),
    size = (600, 800),
)
display(p)
savefig(plotsdir("gaussian", "full_plots_D=$D.png"))
savefig(plotsdir("gaussian", "full_plots_D=$D.svg"))