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
dataset = "spam"
dataset_file = endswith(dataset, ".csv") ? dataset : dataset * ".csv"
data = CSV.read(datadir("exp_raw", "logistic", dataset_file), DataFrame; header=true)
X = Matrix(data[1:end-1])
n_samples, n_dim = size(X)
y = Vector(data[end])

all_results = collect_results(datadir("results", "logistic", dataset))

models = [
    :gpf,
    :gf,
    :dsvi,
    :fcs,
    #:iblr,
]

metrics = [
    :nll_train,
    :nll_test,
    :acc_train,
    :acc_test,
    ]
text_mf = Dict(
        :none => " - No MF",
        :full => " - Full MF",
        :true => " - Full MF"
)

function plot_logistic_convergence(dataset, df, eta=1e-4;
                        show_std_dev = false,
                        show_lgd = true,
                        use_time = false,
    )
    all_res = @linq df |> where(:eta .== eta) |> where(:B .== 100)
    @info "Total of $(nrow(all_res)) for given parameters"
    if nrow(all_res) == 0
        @warn "Results for n_dim=$n_dim, cond=$cond not available yet"
        return nothing
    end
    d_res = Dict()
    # nrow(res) == 1 || error("Number of rows is not unique or is empty")
    for alg in algs
        # d_res[alg] = @linq res |> where(endswith.(:path, Regex("$(alg).*bson")))
        alg_res = @linq all_res |> where(:alg .=== alg) # endswith.(:path, Regex("$(alg).*bson")))
        if nrow(alg_res) > 0
            global vals = first(alg_res.vals)
            if isempty(first(vals).storage)
                continue
            end
            res = Dict()
            res[:mf] = first(alg_res.mf)
            res[:t_m], res[:t_v] = process_time(vals)
            res[:iter] = first(get(vals[1], :t_tic))
            for metric in [:acc_train, :acc_test, :nll_train, :nll_test]
                res[Symbol(metric, "_m")], res[Symbol(metric, "_v")] = get_mean_and_var(vals, metric)
            end
            d_res[alg] = res
        end
    end
    # Plotting
    ylog = true
    # ymin = eps(Float64)
    # ymax = 1e4
    plots = Dict()
    for m in metrics
        plots[m] = plot(
                title=string(m),
                xaxis=:log,
                legend=false,
                xlabel=use_time ? "Time [s]" : "Iterations"
                )
        for alg in algs
            if haskey(d_res, alg)
                res = d_res[alg]
                plot!(
                    use_time ? res[:t_m] : save_times[1:length(res[:t_m])],
                    res[Symbol(m, "_m")],
                    ribbon= show_std_dev ? sqrt.(res[Symbol(m, "_v")]) : nothing,
                    label="",#string(alg, text_mf[res[:mf]])
                )
            end
        end
    end
    plots[:legend] = plot(showaxis=false, hidedecorations=true, grid=false, legendfontsize=10.0, title=dataset)
    for alg in keys(d_res)
        plot!([], [], label=string(alg, text_mf[d_res[alg][:mf]]))
    end    
    return plots
end

ps = plot_logistic_convergence(dataset, all_results;
    show_std_dev = true,
    use_time = true)
plot(values(ps)...)

## Plot results

show_legend = false
use_time = true
show_std = false
plots = Dict()
for m in metric
    plots[m] = plot(
                title=string(m),
                xaxis=:log,
                legend=false,
                xlabel=use_time ? "Time [s]" : "Iterations"
                )
    for alg in keys(results)#models
        res = results[alg]
        if show_std
            plot!(use_time ? res[:t_m] : save_times[1:length(res[:t_m])], res[Symbol(m, "_m")], ribbon=sqrt.(res[Symbol(m, "_v")]), label=string(alg, text_mf[ps[:mf]]))
        else
            plot!(use_time ? res[:t_m] : save_times[1:length(res[:t_m])], res[Symbol(m, "_m")], label=string(alg, text_mf[ps[:mf]]))
        end
    end
end
plots[:legend] = plot(showaxis=false, hidedecorations=true, grid=false, legendfontsize=10.0, title=dataset)
for alg in keys(results)
    plot!([], [], label=string(alg, text_mf[ps[:mf]]))
end
plot(values(plots)...)


