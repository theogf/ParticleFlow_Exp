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
    @unpack dim, n_particles, n_iters, n_runs, cond1, cond2, full_cov = exp_p
    n_particles = iszero(n_particles) ? dim + 1 : n_particles # If nothing is given use dim+1 particlesz`
    μ = sort(randn(dim))
    Σ = if full_cov
        Q, _ = qr(rand(dim, dim)) # Create random unitary matrix
        Λ = Diagonal(10.0 .^ range(-1, 2, length = dim))
        Symmetric(Q * Λ * Q')
    else
        I(dim)
    end
    α = eps(Float64)
    while !isposdef(Σ)
        Σ .+= α * I(dim)
        α *= 10
    end

    # Flux.@functor TuringDenseMvNormal
    d_target = TuringDenseMvNormal(μ, Σ)
    ## Create the model
    function logπ_gauss(θ)
        return logpdf(d_target, θ)
    end

    gpf = Vector{Any}(undef, n_runs)
    gaussflow = Vector{Any}(undef, n_runs)
    advi = Vector{Any}(undef, n_runs)
    steinvi = Vector{Any}(undef, n_runs)

    for i in 1:n_runs
        @info "Run $i/$(n_runs)"
        opt = exp_p[:opt]
        μ_init = randn(dim)
        Σ_init = Matrix(Diagonal(exp.(randn(dim))))
        p_init = MvNormal(μ_init, Σ_init)
        x_init = rand(p_init, n_particles)

        ## Create dictionnaries of parameters
        general_p =
            Dict(:hyper_params => nothing, :hp_optimizer => ADAGrad(0.1), :n_dim => dim, :gpu => false)
        gflow_p = Dict(
            :run => exp_p[:gpf],
            :n_particles => n_particles,
            :max_iters => n_iters,
            :cond1 => cond1,
            :cond2 => cond2,
            :opt => deepcopy(opt),
            :callback => wrap_cb(),
            :mf => false,
            :init => x_init,
        )
        gaussflow_p = Dict(
            :run => exp_p[:gaussf],
            :n_samples => n_particles,
            :max_iters => n_iters,
            :cond1 => cond1,
            :cond2 => cond2,
            :opt => deepcopy(opt),
            :callback => wrap_cb(),
            :mf => false,
            :init => (μ_init, sqrt.(Σ_init)),
        )
        advi_p = Dict(
            :run => exp_p[:advi] && !cond1 && !cond2,
            :n_samples => n_particles,
            :max_iters => n_iters,
            :opt => deepcopy(opt),
            :callback => wrap_cb(),
            :init => (μ_init, sqrt.(Σ_init)),
        )
        stein_p = Dict(
            :run => exp_p[:steinvi] && !cond1 && !cond2,
            :n_particles => n_particles,
            :max_iters => n_iters,
            :kernel => KernelFunctions.transform(SqExponentialKernel(), 1.0),
            :opt => deepcopy(opt),
            :callback => wrap_cb(),
            :init => x_init,
        )

        # Train all models
        _gpf, _gaussvi, _advi, _steinvi =
            train_model(logπ_gauss, general_p, gflow_p, gaussflow_p, advi_p, stein_p)
        gpf[i] = _gpf
        gaussflow[i] = _gaussvi
        advi[i] =  _advi
        steinvi[i] = _steinvi
    end

    file_prefix = savename(exp_p)
    tagsave(datadir("results", "gaussian_v2", file_prefix * ".bson"),
            @dict dim n_particles full_cov n_iters n_runs cond1 cond2 gpf gaussflow advi steinvi exp_p d_target;
            safe=false, storepatch = false)
end
