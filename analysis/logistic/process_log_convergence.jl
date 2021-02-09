using DrWatson
@quickactivate
include(projectdir("analysis", "post_process.jl"))
# include(srcdir("utils", "linear.jl"))
include(srcdir("utils", "tools.jl"))
using AdvancedVI; const AVI = AdvancedVI
using LinearAlgebra
using MLDataUtils
using ValueHistories
save_times = vcat(1:9, 10:5:99, 100:100:999, 1000:1000:9999, 10000:10000:100000)

## Load data
dataset = "ionosphere"
dataset_file = endswith(dataset, ".csv") ? dataset : dataset * ".csv"
data = CSV.read(datadir("exp_raw", "logistic", dataset_file), DataFrame; header=true)
X = Matrix(data[1:end-1])
n_samples, n_dim = size(X)
y = Vector(data[end])
cpalette = :seaborn_colorblind

all_results = collect_results(datadir("results", "logistic"), subfolders=true)

## Cleanup of the data

dir = datadir("results", "logistic", dataset)
ds = []
for f in readdir(dir)
    if isfile(joinpath(dir, f))
        d = BSON.load(joinpath(dir, f))
        if d isa Dict{Any, Any}
            push!(ds, d)
            if haskey(d, :(vals, alg))
                vals, alg = d[:(vals, alg)]
                d[:vals] = vals
                d[:alg] = alg
                delete!(d, :(vals, alg))
                d = Dict{Symbol, Any}(d)
                save(joinpath(dir, f), d)
                @info "WOW"
            end
        end
    end
end




metrics = [
    :nll_train,
    :nll_test,
    :acc_train,
    :acc_test,
    ]
text_mf = Dict(
        :none => "",
        :full => " - MF",
        :true => " - MF"
)

text_natmu = Dict(
        true => "",
        false => "",
)
alg_order = [
    :gpf,
    :gf,
    :dsvi,
    :fcs,
    :iblr,
]

function plot_logistic_convergence(
                        dataset,
                        df,
                        eta=1e-4,
                        n_particles=100,
                        B=-1;
                        show_std_dev = false,
                        show_lgd = true,
                        use_time = false,
                        black_list = [],
                        use_quantile = false,
    )
    all_res = @linq df |> 
                where(:eta .== eta) |> 
                where(:B .== B) |>
                where(:n_particles .== n_particles) |>
                where(:dataset .== dataset)

    @info "Total of $(nrow(all_res)) for given parameters"
    if nrow(all_res) == 0
        @warn "Results for n_dim=$n_dim, cond=$cond not available yet"
        return nothing
    end
    global d_res = Dict()
    # nrow(res) == 1 || error("Number of rows is not unique or is empty")
    for alg in algs[1:end-1]
        # d_res[alg] = @linq res |> where(endswith.(:path, Regex("$(alg).*bson")))
        alg_res = @linq all_res |> where(:alg .=== alg) # endswith.(:path, Regex("$(alg).*bson")))
        @info "Given the parameters there are $(nrow(alg_res)) rows for algorithm $alg"

        for row in eachrow(alg_res)
            vals = row.vals
            if isempty(first(vals).storage)
                continue
            end
            res = Dict()
            res[:mf] = row.mf
            if !isempty(black_list) && (alg, row.mf, row.natmu) ∈ black_list
                @info "Passing $((alg, row.mf, row.natmu))"
                continue
            end
            res[:t_m], res[:t_v] = process_time(vals, Val(alg))
            res[:iter] = first(get(vals[1], :t_tic))
            for metric in [:acc_train, :acc_test, :nll_train, :nll_test]
                res[Symbol(metric, "_m")], res[Symbol(metric, "_v")] = get_mean_and_var(vals, metric, use_quantile=use_quantile)
            end
            d_res[(alg, row.mf, row.natmu)] = res
        end
    end
    # Plotting
    ylog = true
    # ymin = eps(Float64)
    # ymax = 1e4
    plots = Dict()
    ordered_keys = sort(collect(keys(d_res)), by=x->findfirst(y->y==x[1], alg_line_order))
    for m in [:nll_test]#metrics
        plots[m] = plot(
                title=string(m),
                xaxis=:log,
                legend=false,
                xlabel=use_time ? "Time [s]" : "Iterations",
                palette=cpalette,
                )
        for (alg, mf, natmu) in ordered_keys #algs
            res = d_res[(alg, mf, natmu)]
            if use_quantile
                plot!(
                    use_time ? res[:t_m] : save_times[1:length(res[:t_m])],
                    res[Symbol(m, "_m")],
                    fillrange=show_std_dev ? res[Symbol(m, "_v")]' : nothing,
                    fillalpha=0.3,
                    label=string(alg_lab[alg], text_mf[res[:mf]], text_natmu[natmu]),
                    color=alg_col[alg],
                    linestyle=:solid,#alg_mf_line[mf]
                )
            else
                plot!(
                    use_time ? res[:t_m] : save_times[1:length(res[:t_m])],
                    res[Symbol(m, "_m")],
                    ribbon= show_std_dev ? sqrt.(res[Symbol(m, "_v")]) : nothing,
                    fillalpha=0.3,
                    label=string(alg_lab[alg], text_mf[res[:mf]], text_natmu[natmu]),
                    color=alg_col[alg],
                    linestyle=:solid,#alg_mf_line[mf]
                )
            end
        end
        for (alg, mf, natmu) in ordered_keys #algs
            res = d_res[(alg, mf, natmu)]
            plot!(
                use_time ? res[:t_m] : save_times[1:length(res[:t_m])],
                res[Symbol(m, "_m")],
                label="",
                color=alg_col[alg],
                linestyle=alg_ls[alg],
                lw=alg_lw[alg]
            )
        end
    end
    plots[:legend] = plot(showaxis=false, hidedecorations=true, grid=false, legendfontsize=10.0, title=dataset)
    ordered_keys = sort(collect(keys(d_res)), by=x->findfirst(y->y==x[1], algs))
    for (alg, mf, natmu) in ordered_keys
        plot!(
            [],
            [],
            label=string(alg_lab[alg], text_mf[mf], text_natmu[natmu]),
            color=alg_col[alg],
            linestyle=alg_ls[alg],#alg_mf_line[mf],
        )
    end    
    return plots
end

## Plot some stuff 
with_std = true
B = 100
n_particles = 100
eta = 1e-4
all_ps_mf = Dict()
fsize=(600,800)

for dataset in ["ionosphere", "mushroom", "krkp", "spam"]
    # for B in [-1, 100], n_particles in [2, 5, 100]
        ps = plot_logistic_convergence(
            dataset,
            all_results,
            eta,
            n_particles,
            B;
            show_std_dev = with_std,
            use_quantile = true,
            use_time = true,
            black_list = [
                (:gpf, :none, false),
                (:gpf, :none, true),
                # (:gpf, :full, false),
                (:gpf, :full, true),
                (:gpf, :true, true),
                (:gf, :none, false),
                # (:gf, :full, false),
                (:gf, :none, true),
                (:gf, :full, true),
                (:gf, :true, true),
                # (:dsvi, :full, false),
                (:dsvi, :none, false),
                (:fcs, :none, false),
                # (:iblr, :full, true)
                ],
            )
        # full_plot = plot(ps[:legend], getindex.(Ref(ps), metrics)..., layout= @layout [A{0.3w} [B C; D E]])
        # display(full_plot)
        # mkpath(plotsdir("logistic", dataset))
        # savefig(plotsdir("logistic", dataset, savename(@dict(dataset, eta, B, n_particles), "png")))
        all_ps_mf[dataset] = plot(ps[:nll_test], titlefontsize=17.0, title=latexstring("$dataset"), legend=false, ylabel="NLL")
        
        display(ps[:nll_test])
    # end
end
# all_ps_mf[:legend1] = Plots.plot(
#         showaxis=false,
#         legend=lloc,
#         hidedecorations=true,
#         grid=false,
#         legendfontsize=lfsize,
#         fg_legend=:white,
#         bg_legend=:white,
#     )

# for alg in [:gpf, :gf]
#     plot!(
#         all_ps_mf[:legend1],
#         [],
#         [],
#         color=alg_col[alg],
#         label=alg_lab[alg],
#     )
# end
# all_ps_mf[:legend2] = Plots.plot(
#         showaxis=false,
#         legend=lloc,
#         hidedecorations=true,
#         grid=false,
#         legendfontsize=lfsize,
#         fg_legend=:white,
#         bg_legend=:white,
#     )

# for alg in [:dsvi, :iblr]
#     plot!(
#         all_ps_mf[:legend2],
#         [],
#         [],
#         color=alg_col[alg],
#         label=alg_lab[alg],
#     )
# end
xticks = [10^-1, 10^0, 10^1]
plot!(all_ps_mf["ionosphere"], xlabel="", xticks=xticks, size=fsize)
plot!(all_ps_mf["krkp"], ylabel="", xlabel="", xticks=xticks, size=fsize)
plot!(all_ps_mf["mushroom"], xticks=xticks, size=fsize)
plot!(all_ps_mf["spam"], ylabel="", xticks=xticks, size=fsize)
p = plot(
    getindex.(Ref(all_ps_mf), ["ionosphere", "krkp", "mushroom", "spam"])...,
    layout=@layout([A B;C D]),
    dpi=300,
    size=(600,600),
    # size=fsize,
)
display(p)
savefig(plotsdir("logistic", "all_datasets_mf_B=$(B)_N=$(n_particles)"))

### Plot without mf
# B=-1
# n_particles = 100
all_ps = Dict()
for dataset in ["ionosphere", "mushroom", "krkp", "spam"]
    # for B in [-1, 100], n_particles in [2, 5, 100]
        @info "Dataset $dataset"
        ps = plot_logistic_convergence(
            dataset,
            all_results,
            eta,
            n_particles,
            B;
            show_std_dev = true,
            use_quantile=true,
            use_time = true,
            black_list = [
                # (:gpf, :none, false),
                (:gpf, :none, true),
                (:gpf, :full, false),
                (:gpf, :full, true),
                (:gpf, :true, true),
                # (:gf, :none, false),
                (:gf, :none, true),
                (:gf, :full, false),
                (:gf, :full, true),
                (:gf, :true, true),
                # (:dsvi, :none, false),
                (:dsvi, :full, false),
                # (:fcs, :none, false),
                (:iblr, :full, true)
                ],
            )
        # full_plot = plot(ps[:legend], getindex.(Ref(ps), metrics)..., layout= @layout [A{0.3w} [B C; D E]])
        # display(full_plot)
        # mkpath(plotsdir("logistic", dataset))
        # savefig(plotsdir("logistic", dataset, savename(@dict(dataset, eta, B, n_particles), "png")))
        all_ps[dataset] = plot(ps[:nll_test], titlefontsize=17.0, title=latexstring(dataset), legend=false, ylabel="NLL")
        
        display(ps[:nll_test])
    # end
end
# all_ps[:legend1] = Plots.plot(
#         showaxis=false,
#         legend=lloc,
#         hidedecorations=true,
#         grid=false,
#         legendfontsize=lfsize,
#         fg_legend=:white,
#         bg_legend=:white,
#     )

# for alg in [:gpf, :gf]
#     plot!(
#         all_ps[:legend1],
#         [],
#         [],
#         color=alg_col[alg],
#         label=alg_lab[alg],
#     )
# end
# all_ps[:legend2] = Plots.plot(
#         showaxis=false,
#         legend=lloc,
#         hidedecorations=true,
#         grid=false,
#         legendfontsize=lfsize,
#         fg_legend=:white,
#         bg_legend=:white,
#     )

# for alg in [:dsvi, :fcs]
#     plot!(
#         all_ps[:legend2],
#         [],
#         [],
#         color=alg_col[alg],
#         label=alg_lab[alg],
#     )
# end
xticks = [10^-1, 10^0, 10^1]
plot!(all_ps["ionosphere"], xlabel="", xticks=xticks, size=fsize,)
plot!(all_ps["krkp"], ylabel="", xlabel="", xticks=xticks, size=fsize,)
plot!(all_ps["mushroom"], xticks=xticks, size=fsize,)
plot!(all_ps["spam"], ylabel="", xticks=xticks, size=fsize)
p = plot(
    getindex.(Ref(all_ps), ["ionosphere", "krkp", "mushroom", "spam"])...,
    layout=@layout([A B;C D]),
    dpi=300,
    size=(600,600)
)
display(p)
savefig(plotsdir("logistic", "all_datasets_no_mf_B=$(B)_N=$(n_particles)"))
# plot(ps[:nll_test], legend=true)

## Creating a simple legend plot

leg_ps = Dict()
leg_ps[:legend1] = Plots.plot(
        showaxis=false,
        legend=lloc,
        hidedecorations=true,
        grid=false,
        legendfontsize=lfsize,
        fg_legend=:white,
        bg_legend=:white,
    )

for alg in [:gpf, :gf]
    plot!(
        leg_ps[:legend1],
        [],
        [],
        color=alg_col[alg],
        label=alg_lab[alg],
        ls=alg_ls[alg],
        lw=8.0
    )
end
leg_ps[:legend2] = Plots.plot(
        showaxis=false,
        legend=lloc,
        hidedecorations=true,
        grid=false,
        legendfontsize=lfsize,
        fg_legend=:white,
        bg_legend=:white,
    )

for alg in [:dsvi, :fcs]
    plot!(
        leg_ps[:legend2],
        [],
        [],
        color=alg_col[alg],
        label=alg_lab[alg],
        ls=alg_ls[alg]
    )
end

leg_ps[:legend3] = Plots.plot(
        showaxis=false,
        legend=lloc .+ (0.0, 0.2),
        hidedecorations=true,
        grid=false,
        legendfontsize=lfsize,
        fg_legend=:white,
        bg_legend=:white,
    )

for alg in [:iblr]
    plot!(
        leg_ps[:legend3],
        [],
        [],
        color=alg_col[alg],
        label=alg_lab[alg],
    )
end

p = plot(leg_ps[:legend1], leg_ps[:legend2], leg_ps[:legend3], layout=(1,3), dpi=300, size=(500,120))
display(p)
savefig(plotsdir("logistic", "legend.png"))
savefig(plotsdir("logistic", "legend.svg"))