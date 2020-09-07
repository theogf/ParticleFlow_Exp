using DrWatson; @quickactivate
using PDMats, DataFrames, BSON, ValueHistories
using Plots; pyplot()
default(lw=3.0, legendfontsize = 15.0, labelfontsize = 15.0, tickfontsize = 13.0)
using ColorSchemes
colors = ColorSchemes.seaborn_colorblind


## Extract timing
function extract_time(h)
    t_init = get(h, :t_tic)[2]
    t_end = get(h, :t_toc)[2]
    t_diff = cumsum(t_end-t_init)
    t_init[2:end] .-= t_diff[1:end-1]
    return t_init .- get(h, :t_start)[2][1]
end




        err_μ_g = norm.(Ref(μ) .- get(g_h, :mu)[2])
        p_μ = plot(t_g, err_μ_g, label = "Gauss", title = "D = $D", xlabel = "t [s]", ylabel = "|μ - μ'|'", color = colors[1], xaxis = :log)#, yaxis = :log)
        if stein_p[:run]
            err_μ_s = norm.(Ref(μ) .- get(s_h, :mu)[2])
            plot!(t_s, err_μ_s, label = "Stein", color = colors[3])
        end
        if advi_p[:run]
            err_μ_a = norm.(Ref(μ) .- get(a_h, :mu)[2])
            plot!(t_a, err_μ_a, label = "ADVI", color = colors[2])
        end
        display(p_μ)
        savefig(joinpath(@__DIR__, "..", "plots", "gaussian", "mu_D=$(D)_" * (d_target isa DiagNormal ? "diag" : "") * ".png"))

        err_Σ_g = norm.(Ref(Σ) .- reshape.(get(g_h, :sig)[2], D, D))
        p_Σ = plot(t_g, err_Σ_g, label = "Gauss", title = "D = $D", xlabel = "t [s]", ylabel = "|Σ - Σ'|'", color = colors[1], xaxis = :log)#, yaxis = :log, )
        if stein_p[:run]
            err_Σ_s = norm.(Ref(Σ) .- reshape.(get(s_h, :sig)[2], D, D))
            plot!(t_s, err_Σ_s, label = "Stein", color = colors[3])
        end
        if advi_p[:run]
            err_Σ_a = norm.(Ref(Σ) .- reshape.(get(a_h, :sig)[2], D, D))
            plot!(t_a, err_Σ_a, label = "ADVI", color = colors[2])
        end
        display(p_Σ)
        savefig(joinpath(@__DIR__, "..", "plots", "gaussian", "Sigma_D=$(D)_" * (d_target isa DiagNormal ? "diag" : "") * ".png"))
