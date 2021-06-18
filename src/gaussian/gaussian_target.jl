using DataFrames
using BSON
using Flux
using Zygote
include(srcdir("train_model.jl"))
include(srcdir("utils", "tools.jl"))
include(srcdir("utils.jl"))
function run_gaussian_target(exp_p)
    @unpack seed = exp_p
    Random.seed!(seed)
    # AVI.setadbackend(:reversediff)
    AVI.setadbackend(:zygote)

    ## Create target distribution
    @unpack n_dim, n_particles, n_iters, n_runs, natmu, cond, eta, opt_det, opt_stoch, comp_hess = exp_p
    n_particles = iszero(n_particles) ? n_dim + 1 : n_particles # If nothing is given use dim+1 particlesz`
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
    exp_p[:gpf] = opt_stoch == :Descent ? exp_p[:gpf] : false
    exp_p[:dsvi] = natmu ? false : exp_p[:dsvi]
    exp_p[:fcs] = natmu ? false : exp_p[:fcs]
    exp_p[:iblr] = natmu ? exp_p[:iblr] : false
    exp_p[:svgd_linear] = natmu ? false : exp_p[:svgd_linear]
    exp_p[:svgd_rbf] = natmu ? false : exp_p[:svgd_rbf]
    
    ## Create the file prefix for storing the results
    file_prefix = @savename n_iters n_runs n_dim n_particles cond eta
    file_prefix = partial_exp ? joinpath("partial", file_prefix) : file_prefix
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
            elseif alg == :svgd_linear || alg == :svgd_rbf
                @savename(opt_det)
            end
            if isfile(datadir("results", "gaussian", file_prefix * alg_string * ".bson"))
                if filesize(datadir("results", "gaussian", file_prefix * alg_string * ".bson")) > 0
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

    parameters = BSON.load(datadir("exp_raw", "gaussian", @savename(cond, n_dim) * ".bson"))
    @unpack μ_target, Σ_target = parameters
    d_target = MvNormal(μ_target, Σ_target)
    ## Create the model
    function logπ_gauss(θ)
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
        L_init = cholesky(Σ_init).L
        p_init = MvNormal(μ_init, Σ_init)
        x_init = rand(p_init, n_particles)

        ## Create dictionnaries of parameters
        general_p = Dict(
            :hyper_params => nothing,
            :hp_optimizer => nothing,
            :n_dim => n_dim,
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
            :max_iters => n_iters,
            :natmu => natmu,
            :opt => @eval($opt_stoch($eta)),
            :callback => wrap_cb(),
            :mf => false,
            :init => (copy(μ_init), Matrix(L_init)),
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
            :opt => @eval($opt_stoch($eta)),
            :callback => wrap_cb(),
            :mf => false,
            :init => (copy(μ_init), Matrix(L_init - Diagonal(L_init) / sqrt(2)), diag(L_init) / sqrt(2)),
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
        params[:svgd_linear] = Dict(
            :run => exp_p[:svgd_linear],
            :n_particles => n_particles,
            :max_iters => n_iters,
            :opt => @eval($opt_det($eta)),
            :callback => wrap_cb(),
            :mf => false,
            :init => copy(x_init),
        )
        params[:svgd_rbf] = Dict(
            :run => exp_p[:svgd_rbf],
            :n_particles => n_particles,
            :max_iters => n_iters,
            :opt => @eval($opt_det($eta)),
            :callback => wrap_cb(),
            :mf => false,
            :init => copy(x_init),
        )

        # Train all models
        hist, params =
            train_model(logπ_gauss, general_p, params)
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
        elseif alg == :svgd_linear || alg == :svgd_rbf
            @savename(opt_det)
        end
        if exp_p[alg] && mode == :save
            DrWatson.save(
                datadir("results", "gaussian", file_prefix * alg_string * ".bson"),
                merge(exp_p, @dict(vals, alg)))
        end
    end
end
