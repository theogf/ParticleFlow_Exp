using DataFrames
using BSON
using Flux
include(srcdir("train_model.jl"))
include(srcdir("utils", "tools.jl"))
function run_gaussian_target(exp_p)
    @unpack seed = exp_p
    Random.seed!(seed)
    AVI.setadbackend(:reversediff)

    ## Create target distribution
    @unpack n_dim, n_particles, n_iters, n_runs, natmu, cond, eta, opt_det, opt_stoch, comp_hess = exp_p
    n_particles = iszero(n_particles) ? n_dim + 1 : n_particles # If nothing is given use dim+1 particlesz`
    
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
        μ_init = μs_init[i]
        Σ_init = Σs_init[i]
        L_init = cholesky(Σ_init).L
        p_init = MvNormal(μ_init, Σ_init)
        x_init = rand(p_init, n_particles)

        ## Create dictionnaries of parameters
        general_p =
            Dict(:hyper_params => nothing, :hp_optimizer => nothing, :n_dim => n_dim, :gpu => false)
        params = Dict{Symbol, Dict}()
        params[:gpf] = Dict(
            :run => exp_p[:gpf] && (opt_stoch == :Descent),
            :n_particles => n_particles,
            :max_iters => n_iters,
            :natmu => natmu,
            :opt => @eval($opt_det($eta)),
            :callback => wrap_cb(),
            :mf => false,
            :init => copy(x_init),
        )
        params[:gf] = Dict(
            :run => exp_p[:gf] && !natmu,
            :n_samples => n_particles,
            :max_iters => n_iters,
            :natmu => natmu,
            :opt => @eval($opt_stoch($eta)),
            :callback => wrap_cb(),
            :mf => false,
            :init => (copy(μ_init), Matrix(L_init)),
        )
        params[:dsvi] = Dict(
            :run => exp_p[:dsvi] && !natmu,
            :n_samples => n_particles,
            :max_iters => n_iters,
            :opt => @eval($opt_stoch($eta)),
            :callback => wrap_cb(),
            :mf => false,
            :init => (copy(μ_init), deepcopy(L_init)),
        )
        params[:fcs] = Dict(
            :run => exp_p[:fcs] && !natmu,
            :n_samples => n_particles,
            :max_iters => n_iters,
            :opt => @eval($opt_stoch($eta)),
            :callback => wrap_cb(),
            :mf => false,
            :init => (copy(μ_init), Matrix(L_init - Diagonal(L_init) / sqrt(2)), diag(L_init) / sqrt(2)),
        )
        params[:iblr] = Dict(
            :run => exp_p[:iblr] && !natmu,
            :n_samples => n_particles,
            :max_iters => n_iters,
            :opt => Descent(eta),
            :callback => wrap_cb(),
            :comp_hess => comp_hess,
            :mf => false,
            :init => (copy(μ_init), Matrix(inv(Σ_init))),
        )
        params[:svgd] = Dict(
            :run => exp_p[:svgd] && !natmu,
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

    file_prefix = @savename n_iters n_runs n_dim n_particles cond eta
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

        DrWatson.save(
            datadir("results", "gaussian", file_prefix * alg_string * ".bson"),
            merge(exp_p, @dict(vals, d_target)))
    end
end
