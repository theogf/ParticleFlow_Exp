using Turing
using Distributions, DistributionsAD, Bijectors
using AdvancedVI; const AVI = AdvancedVI;
using KernelFunctions, Flux
using ValueHistories
using Parameters
using LinearAlgebra
using Plots
default(lw=2.0)
using ColorSchemes
colors = ColorSchemes.seaborn_colorblind

AVI.setadbackend(:forwarddiff)

function wrap_cb(h::MVHistory)
    return function(i, q, hp)
        if !isnothing(hp)
            cb_hp(h, i, hp)
        end
        cb_var(h, i, q)
        cb_val(h, i, q)
    end
end

function cb_hp(h, i::Int, hp)
    push!(h, :σ_kernel, i, hp[1])
    push!(h, :l_kernel, i, hp[2])
    push!(h, :σ_gaussian, i, hp[3])
end

cb_var(h, i::Int, q::TransformedDistribution) = cb_var(h, i, q.dist)

function cb_var(h, i::Int, q::Union{SamplesMvNormal, SteinDistribution})
    push!(h, :mu, i, copy(mean(q)))
    push!(h, :sig, i, copy(cov(q)[:]))
end

function cb_var(h, i::Int, q::TuringDenseMvNormal)
    push!(h, :mu, i, q.m)
    push!(h, :sig, i, Matrix(q.C)[:])
end

function cb_val(h, i, q)
    return nothing
end


function train_model(X, y, logπ, general_p, gflow_p, advi_p, stein_p)
    ## Initialize algorithms
    gflow_vi, gflow_q = init_gflow(gflow_p, general_p)
    advi_vi, advi_q, advi_init = init_advi(advi_p, general_p)
    stein_vi, stein_q = init_stein(stein_p, general_p)

    ## Set up storage arrays
    gflow_h = MVHistory()
    advi_h = MVHistory()
    stein_h = MVHistory()

    ## Run algorithms
    if !isnothing(gflow_vi)
        @info "Running Gaussian Flow Particles"
        AVI.vi(
            logπ,
            gflow_vi,
            gflow_q,
            optimizer = gflow_p[:opt],
            hyperparams = deepcopy(general_p[:hyper_params]),
            hp_optimizer = deepcopy(general_p[:hp_optimizer]),
            callback = gflow_p[:callback](gflow_h)
        )
    end
    if !isnothing(advi_vi)
        @info "Running ADVI"
        AVI.vi(
            logπ,
            advi_vi,
            advi_q,
            advi_init,
            optimizer = advi_p[:opt],
            hyperparams = deepcopy(general_p[:hyper_params]),
            hp_optimizer = deepcopy(general_p[:hp_optimizer]),
            callback =  advi_p[:callback](advi_h)
        )
    end
    if !isnothing(stein_vi)
        @info "Running Stein VI"
        AVI.vi(
            logπ,
            stein_vi,
            stein_q,
            optimizer = stein_p[:opt],
            hyperparams = deepcopy(general_p[:hyper_params]),
            hp_optimizer = deepcopy(general_p[:hp_optimizer]),
            callback = stein_p[:callback](stein_h)
        )
    end
    return gflow_h, advi_h, stein_h
end


function save_histories(gflow_h, advi_h, stein_h, general_p)
    names = ("gauss", "advi", "stein")
    for (h, name) in zip((gflow_h, advi_h, stein_h), names)
        if length(keys(h)) != 0
            @info "Saving histories of algorithm $name"
            save_results(h, name, general_p)
        end
    end
end

function save_results(h, name, general_p)

end

function init_gflow(gflow_p, general_p)
    n_dim = general_p[:n_dim]
    gflow_vi = if gflow_p[:run]
        AVI.PFlowVI(gflow_p[:max_iters], gflow_p[:cond1], gflow_p[:cond2])
    else
        nothing
    end
    isnothing(gflow_p[:init]) ||
        size(gflow_p[:init]) == (n_dim, gflow_p[:n_particles]) # Check that the size of the inital particles respect the model
    gflow_q = SamplesMvNormal(
        isnothing(gflow_p[:init]) ?
            rand(MvNormal(ones(n_dim)), gflow_p[:n_particles]) :
            gflow_p[:init],
    )

    return gflow_vi, gflow_q
end

function init_advi(advi_p, general_p)
    n_dim = general_p[:n_dim]
    advi_vi = if advi_p[:run]
        AVI.ADVI(advi_p[:n_samples], advi_p[:max_iters])
    else
        return nothing, nothing, nothing
    end
    mu_init, L_init =
        isnothing(advi_p[:init]) ? (zeros(n_dim), Matrix(I(n_dim))) : advi_p[:init] # Check that the size of the inital particles respect the model
    advi_q = AVI.transformed(
        TuringDenseMvNormal(mu_init, L_init * L_init'),
        AVI.Bijectors.Identity{1}(),
    )
    return advi_vi, advi_q, vcat(mu_init, L_init[:])
end

function init_stein(stein_p, general_p)
    n_dim = general_p[:n_dim]
    stein_vi = if stein_p[:run]
        AVI.SteinVI(stein_p[:max_iters], stein_p[:kernel])
    else
        nothing
    end
    isnothing(stein_p[:init]) ||
        size(stein_p[:init]) == (n_dim, stein_p[:n_particles]) # Check that the size of the inital particles respect the model
    stein_q = AVI.SteinDistribution(
        isnothing(stein_p[:init]) ?
            rand(MvNormal(ones(n_dim)), stein_p[:n_particles]) :
            stein_p[:init],
    )

    return stein_vi, stein_q
end
