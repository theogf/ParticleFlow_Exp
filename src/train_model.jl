using AdvancedVI;
const AVI = AdvancedVI;
using BlockDiagonals
using Flux
using Distributions
using Bijectors: TransformedDistribution
using KernelFunctions
using ValueHistories
using LinearAlgebra
using ReverseDiff
using StatsBase
using Random

Flux.@functor TransformedDistribution
include(joinpath("utils", "callback.jl"))
include(joinpath("utils", "tools.jl"))
# include(joinpath("utils", "bnn.jl"))
# Main function, take dicts of parameters
# run the inference and return MVHistory objects for each alg.
no_run = Dict(:run => false)
algs = [
    :gpf,
    :gf,
    :dsvi,
    :fcs,
    :iblr,
    :svgd_linear,
    :svgd_rbf,
]


function train_model(logπ, general_p, params)
    ## Initialize algorithms
    algs = [
        :gpf,
        :gf,
        :dsvi,
        :fcs,
        :iblr,
        :svgd_linear,
        :svgd_rbf,
    ]
    vi_alg = Dict{Symbol,Any}()
    q = Dict{Symbol,Any}()
    h = Dict{Symbol,Any}()
    device = general_p[:gpu] ? gpu : cpu
    mode = get!(general_p, :mode, :save)

    for alg in algs
        # Initialize setup
        vi_alg[alg], q[alg] = init_alg(Val(alg), params[alg], general_p)
        h[alg] = MVHistory()
        ## Run algorithm
        if !isnothing(vi_alg[alg])
            try
                @info "Running $(AVI.alg_str(vi_alg[alg]))"
                push!(h[alg], :t_start, Float64(time_ns()) / 1e9)
                AVI.vi(
                    logπ,
                    vi_alg[alg],
                    q[alg] |> device,
                    optimizer = params[alg][:opt] |> device,
                    hyperparams = deepcopy(general_p[:hyper_params]),
                    hp_optimizer = deepcopy(general_p[:hp_optimizer]),
                    callback = params[alg][:callback](h[alg]),
                )
                if mode == :display
                    @info "Alg. $alg :\nmu = $(mean(q[alg])),\ndiag_sig = $(var(q[alg]))"
                end
            catch err
                if err isa InterruptException || !get!(general_p, :unsafe, false)
                    rethrow(err)
                end
            end
        end
    end
    return h, params
end

# Allows to save the histories into a desired file
function save_histories(h, general_p)
    for (alg, hist) in h
        if length(keys(hist)) != 0
            @info "Saving histories of algorithm $alg"
            save_results(hist, alg, general_p)
        end
    end
end

# Initialize distribution and algorithm for Gaussian Particles model
function init_alg(::Val{:gpf}, params, general_p)
    n_dim = general_p[:n_dim]
    alg_vi = if params[:run]
        AVI.GaussPFlow(params[:max_iters], params[:natmu], false)
    else
        return nothing, nothing
    end
    isnothing(params[:init]) || size(params[:init]) == (n_dim, params[:n_particles]) # Check that the size of the inital particles respect the model
    alg_q = if params[:mf] isa AbstractVector
        BlockMFSamplesMvNormal(
            isnothing(params[:init]) ?
                rand(MvNormal(ones(n_dim)), params[:n_particles]) : params[:init],
            params[:mf],
        )
    elseif params[:mf] == Inf
        MFSamplesMvNormal(
            isnothing(params[:init]) ?
                rand(MvNormal(ones(n_dim)), params[:n_particles]) : params[:init]
        )
    else
        SamplesMvNormal(
            isnothing(params[:init]) ?
                rand(MvNormal(ones(n_dim)), params[:n_particles]) : params[:init],
        )
    end

    return alg_vi, alg_q # Return alg. and distr.
end

# Initialize distribution and algorithm for Gaussian Particles model
function init_alg(::Val{:gf}, params, general_p)
    n_dim = general_p[:n_dim]
    alg_vi = if params[:run]
        AVI.GaussFlow(
            general_p[:gpu] ? CUDA.CURAND.default_rng() : Random.GLOBAL_RNG,
            general_p[:gpu] ? gpu : cpu,
            params[:max_iters],
            params[:n_samples],
            params[:natmu],
            false,
        )
    else
        return nothing, nothing
    end
    alg_q = if params[:mf] isa AbstractVector
        AVI.BlockMFLowRankMvNormal(
            (isnothing(params[:init]) ?
                rand(MvNormal(ones(n_dim)), params[:n_samples]) :
                params[:init])...,
            params[:mf],
        )
    elseif params[:mf] == Inf
        AVI.MFMvNormal(
            (isnothing(params[:init]) ?
                (zeros(n_dim), ones(n_dim)) :
                params[:init])...,
        )
    else
        AVI.LowRankMvNormal(
            (isnothing(params[:init]) ?
                (zeros(n_dim), Matrix{Float32}(I(n_dim))) :
                params[:init])...
        )
    end

    return alg_vi, alg_q # Return alg. and distr.
end

# Initialize distribution and algorithm for ADVI model
function init_alg(::Val{:dsvi}, params, general_p)
    n_dim = general_p[:n_dim]
    alg_vi = if params[:run]
        AVI.DSVI(
            general_p[:gpu] ? CUDA.CURAND.default_rng() : Random.GLOBAL_RNG,
            general_p[:gpu] ? gpu : cpu,
            params[:max_iters],
            params[:n_samples],
            )
    else
        return nothing, nothing
    end
    alg_q = if params[:mf] == Inf
        AVI.MFMvNormal(
            (isnothing(params[:init]) ?
                (zeros(n_dim), ones(n_dim)) :
                params[:init])...
        )
    else
        AVI.CholMvNormal(
            (isnothing(params[:init]) ?
                (zeros(n_dim), cholesky(I(n_dim)).L) :
                params[:init])...
        )
    end

    return alg_vi, alg_q # Return alg. and distr.
end

function init_alg(::Val{:fcs}, params, general_p)
    n_dim = general_p[:n_dim]
    alg_vi = if params[:run]
        AVI.FCS(params[:max_iters], params[:n_samples])
    else
        return nothing, nothing
    end
    !isa(params[:mf], AbstractVector) || params[:mf] != Inf || error("FCS cannot be used with Mean-Field")
    alg_q = AVI.FCSMvNormal(
            (isnothing(params[:init]) ?
            (zeros(n_dim), cholesky(I(n_dim)).L / sqrt(2), ones(n_dim) / sqrt(2)) :
            params[:init])...
        )

    return alg_vi, alg_q # Return alg. and distr.
end

function init_alg(::Val{:iblr}, params, general_p)
    n_dim = general_p[:n_dim]
    alg_vi = if params[:run]
        AVI.IBLR(params[:max_iters], params[:n_samples], params[:comp_hess])
    else
        return nothing, nothing
    end
    alg_q = if params[:mf] == Inf
        AVI.DiagPrecisionMvNormal(
            (isnothing(params[:init]) ?
                (zeros(n_dim), ones(n_dim)) :
                params[:init])...
        )
    else
        AVI.PrecisionMvNormal(
            (isnothing(params[:init]) ?
                (zeros(n_dim), Matrix{Float32}(I(n_dim))) :
                params[:init])...
        )
    end

    return alg_vi, alg_q # Return alg. and distr.
end

function init_alg(svgd::Union{Val{:svgd_linear}, Val{:svgd_rbf}}, params, general_p)
    n_dim = general_p[:n_dim]
    kernel = if svgd isa Val{:svgd_linear}
        :linear
    elseif svgd isa Val{:svgd_rbf}
        :rbf 
    end
    alg_vi = if params[:run]
        if kernel == :linear
            AVI.SVGD(
                general_p[:gpu] ? gpu : cpu,
                LinearKernel(;c=1),
                params[:max_iters],
            )
        else
            AVI.SVGD(
                general_p[:gpu] ? gpu : cpu,
                general_p[:gpu] ? gpu(compose(SqExponentialKernel(), GPUScaleTransform(1.0))) : compose(SqExponentialKernel(), ScaleTransform(1.0)),
                params[:max_iters],
            )
        end
    else
        return nothing, nothing
    end
    isnothing(params[:init]) || size(params[:init]) == (n_dim, params[:n_particles]) # Check that the size of the inital particles respect the model
    alg_q = AVI.EmpiricalDistribution(
                isnothing(params[:init]) ?
                rand(MvNormal(ones(n_dim)), params[:n_particles]) : 
                params[:init],
        )
    return alg_vi, alg_q # Return alg. and distr.
end