using DataFrames
using BSON
using Flux
using Zygote
using PDMats
include(srcdir("train_model.jl"))
include(srcdir("utils", "tools.jl"))
function run_lowrank_target(exp_p)
    @unpack seed = exp_p
    Random.seed!(seed)
    # AVI.setadbackend(:reversediff)
    AVI.setadbackend(:zygote)

    ## Create target distribution
    @unpack n_iters, n_runs, natmu, K, dof, eta, opt_det, opt_stoch, comp_hess = exp_p
    n_particles = K + 1
    partial_exp = get!(exp_p, :partial, false)
    mode = get!(exp_p, :mode, :save)
    ## Adapt the callback function given the experiment
    f = if partial_exp
        function (h::MVHistory)
            return function (i, q, hp)
                if i == (n_iters - 1)
                    cb_tic(h, i)
                    push!(h, :x, i, q.dist.x)
                    cb_toc(h, i)
                end
            end
        end
    else
        wrap_cb()
    end

    ## Adapt the running given the setup:
    # exp_p[:gpf] = opt_det == :Descent ? exp_p[:gpf] : false
    exp_p[:dsvi] = natmu ? false : exp_p[:dsvi]
    exp_p[:fcs] = natmu ? false : exp_p[:fcs]
    exp_p[:svgd] = natmu ? false : exp_p[:svgd]
    
    ## Create the file prefix for storing the results
    file_prefix = @savename n_iters n_runs n_particles K dof eta
    ## Check that the simulation has not been run already
    for alg in algs
        if exp_p[alg]
            alg_string = "_" * string(alg) * "_" * 
            if alg == :gpf
                @savename(natmu, opt_det)
            elseif alg == :gf
                @savename(natmu, opt_stoch)
            elseif alg == :dsvi || alg == :fcs
                @savename(opt_stoch)
            elseif alg == :iblr
                @savename(comp_hess)
            elseif alg == :svgd
                @savename(opt_det)
            end
            if isfile(datadir("results", "lowrank", file_prefix * alg_string * ".bson"))
                if filesize(datadir("results", "lowrank", file_prefix * alg_string * ".bson")) > 0
                    if !get!(exp_p, :overwrite, false)
                        @warn "Simulation has been run already - Passing simulation"
                        exp_p[alg] = false
                    else
                        @warn "Simulation has been run already - Overwriting the simulation"
                    end
                end
            end
        end
    end

    parameters = BSON.load(datadir("exp_raw", "lowrank", @savename(K) * ".bson"))
    @unpack μ_target, Σ_target = parameters
    d_target = MvTDist(dof, μ_target, PDMat(Σ_target))
    ## Create the model
    function logπ_lowrank_st(θ)
        return logpdf(d_target, θ)
    end

    hists = Vector{Dict}(undef, n_runs)

    @unpack μs_init, Σs_init = parameters
    for i in 1:n_runs
        @info "Run $i/$(n_runs)"
        if mode == :display
            @warn "Target has:\nmean $(mean(d_target))\nvar=$(var(d_target))"
        end
        μ_init = μs_init[i]
        Σ_init = Σs_init[i]
        L_init = cholesky(Σ_init).U
        L_init_LR = Matrix(L_init)[:, 1:K]
        L_init_LR_less_diag = Matrix(L_init - Diagonal(L_init) / sqrt(2))[:, 1:K]
        p_init = MvNormal(μ_init, Σ_init)
        x_init = rand(p_init, n_particles)

        ## Create dictionnaries of parameters
        general_p = Dict(
            :hyper_params => nothing,
            :hp_optimizer => nothing,
            :n_dim => 20,
            :gpu => false,
            :mode => mode,
        )
        params = Dict{Symbol, Dict}()
        params[:gpf] = Dict(
            :run => exp_p[:gpf],
            :n_particles => n_particles,
            :max_iters => n_iters,
            :natmu => natmu,
            :opt => @eval($opt_det($eta)),
            :callback => f,
            :mf => false,
            :init => copy(x_init),
        )
        params[:gf] = Dict(
            :run => exp_p[:gf],
            :n_samples => n_particles,
            :rank => K,
            :max_iters => n_iters,
            :natmu => natmu,
            :opt => @eval($opt_stoch($eta)),
            :callback => wrap_cb(),
            :mf => false,
            :init => (copy(μ_init), copy(L_init_LR)),
        )
        params[:dsvi] = Dict(
            :run => exp_p[:dsvi],
            :n_samples => n_particles,
            :max_iters => n_iters,
            :opt => @eval($opt_stoch($eta)),
            :callback => wrap_cb(),
            :mf => false,
            :init => (copy(μ_init), deepcopy(L_init)),
        )
        params[:fcs] = Dict(
            :run => exp_p[:fcs],
            :n_samples => n_particles,
            :max_iters => n_iters,
            :rank => K,
            :opt => @eval($opt_stoch($eta)),
            :callback => wrap_cb(),
            :mf => false,
            :init => (copy(μ_init), L_init_LR_less_diag, diag(L_init) / sqrt(2)),
        )
        params[:iblr] = Dict(
            :run => exp_p[:iblr],
            :n_samples => n_particles,
            :max_iters => n_iters,
            :opt => Descent(eta),
            :callback => wrap_cb(),
            :comp_hess => comp_hess,
            :mf => false,
            :init => (copy(μ_init), Matrix(inv(Σ_init))),
        )
        params[:svgd] = Dict(
            :run => exp_p[:svgd],
            :n_particles => n_particles,
            :max_iters => n_iters,
            :opt => @eval($opt_det($eta)),
            :callback => wrap_cb(),
            :mf => false,
            :init => copy(x_init),
        )
        # Train all models
        hist, params =
            train_model(logπ_lowrank_st, general_p, params)
        hists[i] = hist
    end

    ## Save the results if any
    for alg in algs
        vals = [hist[alg] for hist in hists]
        alg_string = "_" * string(alg) * "_" * 
        if alg == :gpf
            @savename(natmu, opt_det)
        elseif alg == :gf
            @savename(natmu, opt_stoch)
        elseif alg == :dsvi || alg == :fcs
            @savename(opt_stoch)
        elseif alg == :iblr
            @savename(comp_hess)
        elseif alg == :svgd
            @savename(opt_det)
        end
        if exp_p[alg] && mode == :save
            DrWatson.save(
                datadir("results", "lowrank", file_prefix * alg_string * ".bson"),
                merge(exp_p, @dict(vals, alg)))
        end
    end
end
