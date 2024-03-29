using DrWatson
@quickactivate
include(projectdir("analysis", "post_process.jl"))


## Load data
text_natmu = Dict(
        true => " - NG",
        false => "",
)

lowrank_algs = [
    :gpf,
    :gf,
    :fcs,
    :svgd_linear,
    :svgd_rbf,
]

lowrank_alg_line_order = [:fcs, :svgd_linear, :svgd_rbf, :gf, :gpf]
lowrank_alg_line_order_dict = Dict(x=>i for (i,x) in enumerate(lowrank_alg_line_order))

## Treat one convergence file
function plot_lowrank(
    K, 
    eta=1e-2;
    show_std_dev = false,
    show_lgd = true,
    use_quantile = true,
    dof = 5.0,
    n_particles=20,
)
    all_res = collect_results(datadir("results", "lowrank", @savename(K)));
    res = @linq all_res |>
        where(:K .== K) |>
        where(:eta .== eta) |>
        where(:n_iters .>= 1000) |>
        where(:dof .== dof) |>
        where(:n_particles .== n_particles) #|>
        # where(:n_runs .== 10)
    @info "Total of $(nrow(res)) for given parameters"
    if nrow(res) == 0
        @warn "Results for K=$K not available yet"
        return nothing, nothing, nothing
    end
    d_res = Dict()
    for alg in lowrank_algs
        d_res[alg] = @linq res |> where(:alg .=== alg) # endswith.(:path, Regex("$(alg).*bson")))
    end
    params_truth = BSON.load(datadir("exp_raw", "lowrank", savename(@dict(K), "bson")))
    truth = MvNormal(params_truth[:μ_target], PDMat(params_truth[:Σ_target]))
    # truth = MvTDist(dof, params_truth[:μ_target], PDMat(params_truth[:Σ_target]))
    # Plotting
    
    ylog = :log
    ymin = eps(Float64)
    ymax = 1e2
    tfsize = 21.0
    p_μ = Plots.plot(
        title = K == 10 ? L"\|m^t - \mu\|" : "",
        titlefontsize = tfsize,
        xlabel = K == 30 ? "Time [s]" : "",
        ylabel = "",
        xaxis = :log,
        # ylims = (ymin, ymax),
        yaxis = ylog,# ? (!show_std_dev ? :log : :linear) : :linear,
        legend = false,
    )
    annotate!(p_μ, 1e-1, 10^(1.2), Plots.text(latexstring("K = $(K)"), :left, 18))
    p_Σ = Plots.plot(
        title = K == 10 ? L"\|C^t- \Sigma\|" : "",
        titlefontsize = tfsize,
        xlabel = K == 30 ? "Time [s]" : "",
        ylabel = "",
        xaxis = :log,
        # ylims = (ymin, ymax),
        yaxis = ylog,# ? (!show_std_dev ? :log : :linear) : :linear,
        legend = false,
    )
    p_legend = Plots.plot(
        showaxis=false,
        hidedecorations=true,
        grid=false,
        legendfontsize=10.0
    )
    p_title = plot(title="K=$K", grid=false, showaxis=false)

    for (i, alg) in enumerate(lowrank_alg_line_order)
        @info "Processing $(alg)"
        d = d_res[alg]
        for row in eachrow(d)
            if alg != :gpf
                # continue
            end
            vals = row.vals 
            if alg == :gf && row.natmu == true
                continue
            elseif alg ∈ [:gf, :dsvi, :fcs] && row.opt_stoch != :RMSProp
                continue
            elseif alg ∈ (:svgd_linear, :svgd_rbf) && row.opt_det != :DimWiseRMSProp
               continue
            # elseif alg == :gpf && row.opt_det != :DimWiseRMSProp
                # continue
            elseif alg == :iblr && row.comp_hess == :rep
                continue
            end
            m, m_v = process_means(vals, mean(truth), use_quantile=use_quantile)
            C, C_v = process_fullcovs(vals, vec(cov(truth)), use_quantile=use_quantile)
            t, _ = process_time(vals, Val(alg))
            if use_quantile
                Plots.plot!(
                    p_μ,
                    t,
                    m,
                    fillrange=show_std_dev ? m_v : nothing,
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
    return p, p_μ, p_Σ
end
mkpath(plotsdir("lowrank"))
plt = Dict()
Ks = [2, 5, 10, 20]
Ks = [10, 20, 30]
η = 0.01
for K in Ks
    plt[K] = Dict()
    p, plt[K][:μ], plt[K][:Σ] = plot_lowrank(K, η; n_particles=20, show_std_dev=true, show_lgd=false, use_quantile=true)
    try
        display(p)
    catch e
        @warn "Plot was empty for K=$K or there was an error"
        continue
    end
    !isnothing(p) ? savefig(plotsdir("lowrank", savename(@dict(K), ".png"))) : nothing
end
## Working with the plots 
lloc = :best
lfsize = 14.0
p_legend1 = Plots.plot(
        showaxis=false,
        legend=lloc,
        hidedecorations=true,
        grid=false,
        legendfontsize=lfsize,
        fg_legend=:white,
        bg_legend=:white,
        legendtitle="Particle Methods",
        legendtitlefontsize=lfsize+1,
        margin=0px
    )
for alg in vcat(lowrank_algs[1], lowrank_algs[end-1:end])
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
        legendtitle="Stochastic Methods",
        legendtitlefontsize=lfsize+1,
    )
for alg in lowrank_algs[2:3]
    plot!(
        p_legend2,
        [],
        [],
        color=alg_col[alg],
        label=alg_lab[alg],
    )
end


p = plot(
    plt[10][:μ], plt[10][:Σ], plt[20][:μ], plt[20][:Σ], plt[30][:μ], plt[30][:Σ], p_legend1, p_legend2;
    dpi = 300,
    layout = @layout([A B;C D;E F;G{0.22h} H]),
    size = (600, 800),
)

display(p)
savefig(plotsdir("lowrank", "full_plots_η=$η.png"))
savefig(plotsdir("lowrank", "full_plots_η=$η.svg"))