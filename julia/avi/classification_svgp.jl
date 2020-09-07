include("train_model.jl")
using StatsFuns: logistic

struct TimeGate
    t_start::Int
    state::IdDict
end

TimeGate(t::Int = 0) = TimeGate(t, IdDict())

function Flux.Optimise.apply!(o::TimeGate, x, Δ)
  n = get!(o.state, x, 1)
  o.state[x] = n + 1
  if n >= o.t_start
      return Δ
  else
      return zero(Δ)
  end
end

function run_svgp_bin(exp_p)
    n_iters = exp_p[:n_iters]
    n_runs = exp_p[:n_runs]
    data = exp_p[:data]
    B = exp_p[:n_batch]
    M = exp_p[:n_ind_points]
    n_particles = exp_p[:n_particles]
    t_gate = exp_p[:t_gate]

    t_gate < n_iters || error("Time gate smaller than total iterations")

    Random.seed!(exp_p[:seed])
    cond1 = exp_p[:cond1]
    cond2 = exp_p[:cond2]

    X, y, θ = if data == "toydata"
        ## Create some toy data
        N = 200
        x = range(0, 1, length = N)
        Z = range(0, 1, length = M)
        θ = vcat(Z, log.([1.0, 10.0]))
        k = exp(θ[1]) * KernelFunctions.transform(SqExponentialKernel(), exp(θ[2]))
        K = kernelmatrix(k, x) + 1e-5I
        f = rand(MvNormal(K))
        likelihood(f) = Bernoulli(logistic(f))
        y = rand(Product(likelihood.(f)))
        x, y, θ
    else
        # Load some data appropriately
        # Banana dataset for starters ?
    end
    n_train = length(y)
    ratio = n_train / B


    function svgp_loss(d, S, P, θ, z)
        kldivergence = logpdf(d, z)
        log_lik = ratio * loglikelihood(Product(Bernoulli.(logistic.(P[S, :] * z))), y[S])
    end

    ## Create the model
    function meta_logπ(θ)
        Z = θ[1:M]
        k = exp(θ[M+1]) * KernelFunctions.transform(SqExponentialKernel(), exp(θ[M+2]))
        Ku = kernelmatrix(k, Z) + 1e-5I
        Kf = kerneldiagmatrix(k, x) .+ 1e-5
        Kfu = kernelmatrix(k, x, Z)
        P = Kfu / Ku
        d = TuringDenseMvNormal(zeros(length(Z)), Ku)
        return function(z)
            S = sample(1:n_train, B, replace=false) # Sample a minibatch
            return svgp_loss(d, S, P, θ, z)
        end
    end
    k = exp(θ[M+1]) * KernelFunctions.transform(SqExponentialKernel(), exp(θ[M+2]))
    Ku = kernelmatrix(k, Z) + 1e-5I
    Kf = kerneldiagmatrix(k, x) .+ 1e-5
    Kfu = kernelmatrix(k, x, Z)
    P = Kfu / Ku
    logπ_reduce = meta_logπ(θ)
    logπ_reduce(rand(M))
    # AVI.setadbackend(:reversediff)
    ## Start experiment
    hp_init = θ .- 1
    hp_init = nothing
    vals = []

    for i in 1:n_runs
        @info "Run $i/$(n_runs)"
        opt = [ADAGrad(0.2), Flux.Optimise.Optimiser(ADAGrad(0.2), TimeGate(t_gate))]

        general_p =
            Dict(:hyper_params => hp_init, :hp_optimizer => ADAGrad(0.1), :n_dim => M)
        gflow_p = Dict(
            :run => true,
            :n_particles => n_particles,
            :max_iters => n_iters,
            :cond1 => cond1,
            :cond2 => cond2,
            :opt => deepcopy(opt),
            :callback => wrap_cb,
            :init => nothing,
        )
        advi_p = Dict(
            :run => false,
            :n_samples => n_particles,
            :max_iters => n_iters,
            :opt => deepcopy(opt),
            :callback => wrap_cb,
            :init => nothing,
        )
        stein_p = Dict(
            :run => false,
            :n_particles => n_particles,
            :max_iters => n_iters,
            :kernel => KernelFunctions.transform(SqExponentialKernel(), 1.0),
            :opt => deepcopy(opt),
            :callback => wrap_cb,
            :init => nothing,
        )


        g_h, a_h, s_h =
            # train_model(x, y, logπ_reduce, general_p, gflow_p, advi_p, stein_p)
            train_model(meta_logπ, general_p, gflow_p, advi_p, stein_p)
        push!(vals, g_h)
    end
    file_prefix = @savename n_particles B M n_iters n_runs

    tagsave(datadir("results", "svgp", "data", file_prefix * ".bson"),
            @dict n_particles B M n_iters n_runs cond1 cond2 vals exp_p;
            safe=false, storepatch = false)
end
