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
    @unpack dim, n_particles, n_iters, n_runs, natmu, full_cov, eta, opt_det, opt_stoch, comp_hess = exp_p
    n_particles = iszero(n_particles) ? dim + 1 : n_particles # If nothing is given use dim+1 particlesz`
    Σ =  get!(exp_p, :Σ, nothing)
    μ = get!(exp_p, :μ, nothing)
    μ = isnothing(μ) ? sort(randn(dim)) : μ
    Σ = if isnothing(Σ)
        if full_cov
            Q, _ = qr(rand(dim, dim)) # Create random unitary matrix
            Λ = Diagonal(10.0 .^ range(-1, 2, length = dim))
            Symmetric(Q * Λ * Q')
        else
            I(dim)
        end
    else
        Σ
    end
    α = eps(Float64)
    while !isposdef(Σ)
        Σ .+= α * I(dim)
        α *= 10
    end

    d_target = TuringDenseMvNormal(μ, Σ)
    ## Create the model
    function logπ_gauss(θ)
        return logpdf(d_target, θ)
    end

    hists = Vector{Dict}(undef, n_runs)

    for i in 1:n_runs
        @info "Run $i/$(n_runs)"
        μ_init = randn(dim)
        Q, _ = qr(rand(dim, dim)) # Create random unitary matrix
        Λ = Diagonal(exp.(randn(dim)))
        Σ_init = Symmetric(Q * Λ * Q')
        L_init = cholesky(Σ_init).L
        p_init = MvNormal(μ_init, Σ_init)
        x_init = rand(p_init, n_particles)

        ## Create dictionnaries of parameters
        general_p =
            Dict(:hyper_params => nothing, :hp_optimizer => nothing, :n_dim => dim, :gpu => false)
        params = Dict{Symbol, Dict}()
        params[:gpf] = Dict(
            :run => exp_p[:gpf],
            :n_particles => n_particles,
            :max_iters => n_iters,
            :natmu => natmu,
            :opt => opt_det(eta),
            :callback => wrap_cb(),
            :mf => false,
            :init => x_init,
        )
        params[:gf] = Dict(
            :run => exp_p[:gf] && !natmu,
            :n_samples => n_particles,
            :max_iters => n_iters,
            :natmu => natmu,
            :opt => opt_stoch(eta),
            :callback => wrap_cb(),
            :mf => false,
            :init => (copy(μ_init), Matrix(L_init)),
        )
        params[:dsvi] = Dict(
            :run => exp_p[:dsvi] && !natmu,
            :n_samples => n_particles,
            :max_iters => n_iters,
            :opt => opt_stoch(eta),
            :callback => wrap_cb(),
            :mf => false,
            :init => (copy(μ_init), deepcopy(L_init)),
        )
        params[:fcs] = Dict(
            :run => exp_p[:fcs] && !natmu,
            :n_samples => n_particles,
            :max_iters => n_iters,
            :opt => opt_stoch(eta),
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

        # Train all models
        hist, params =
            train_model(logπ_gauss, general_p, params)
        hists[i] = hist
    end

    file_prefix = savename(exp_p)
    for alg in algs
        vals = [hist[alg] for hist in hists]
        DrWatson.save(
            datadir("results", "gaussian", file_prefix * string(alg) * ".bson"),
            @dict vals dim n_particles full_cov n_iters n_runs natmu exp_p d_target;
                safe=false, storepatch = false)
    end
end
