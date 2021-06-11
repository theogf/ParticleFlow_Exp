function max_ps_ids(X)
    maxs = findmax.(eachcol(X))
    return first.(maxs), last.(maxs)
end

function conf_and_acc(preds)
    conf_and_acc(max_ps_ids(preds)...)
end
function conf_and_acc(ps, ids)
    s = sortperm(ps)
    bins = [s[i*N20+1:(i+1)*N20] for i in 0:19]
    conf = mean.(getindex.(Ref(ps), bins))
    acc = [mean(ids[bins[i]] .== Flux.onecold(y_test)[bins[i]]) for i in 1:20]
    return conf, acc
end

function extract_info(alg::Union{Val{:gpf},Val{:svgd_linear},Val{:svgd_rbf}}, alg_dir, mf, exp_params)
    @unpack L, eta, batchsize, n_iter, natmu, opt_det, opt_stoch, α, σ_init = exp_params
    target_dir = joinpath(alg_dir, @savename L eta batchsize mf n_iter natmu opt_det opt_stoch α σ_init)
    if !isdir(target_dir)
        return nothing, nothing, nothing, nothing
    end
    res = collect_results(target_dir)
    final_particles = first(res.particles[res.i .== n_iter - 1])
    @infiltrate alg isa Val{:svgd_linear}
    preds = []
    @showprogress for θ in eachcol(final_particles)
        pred = nn_forward(X_test, θ)
        push!(preds, cpu(Flux.softmax(pred)))
    end
    mean_preds = mean(preds)
    nll = Flux.Losses.crossentropy(mean_preds, y_test)
    acc = mean(Flux.onecold(mean_preds) .== Flux.onecold(y_test))
    conf, conf_acc = conf_and_acc(mean_preds)
    return nll, acc, conf, conf_acc
end

function extract_info(::Union{Val{:gf},Val{:dsvi}}, alg_dir, mf, exp_params)
    @unpack L, batchsize, eta, n_iter, natmu, opt_det, opt_stoch, α, σ_init = exp_params
    target_dir = joinpath(alg_dir, @savename L batchsize mf n_iter natmu opt_det opt_stoch eta α σ_init)
    if !isdir(target_dir)
        return nothing, nothing, nothing, nothing
    end
    res = collect_results(target_dir)
    final_q = first(res.q[res.i .== n_iter - 1])
    preds = []
    @showprogress for θ in eachcol(rand(final_q, n_MC))
        pred = nn_forward(X_test, θ)
        push!(preds, cpu(Flux.softmax(pred)))
    end
    mean_preds = mean(preds)
    nll = Flux.Losses.crossentropy(mean_preds, y_test)
    acc = mean(Flux.onecold(mean_preds) .== Flux.onecold(y_test))
    conf, conf_acc = conf_and_acc(mean_preds)
    return nll, acc, conf, conf_acc
end
function extract_info(::Val{:swag}, alg_dir, mf, exp_params)
    @unpack L, batchsize, n_iter, natmu, opt_det, opt_stoch, α, σ_init = exp_params
    target_dir = joinpath(alg_dir, @savename L batchsize mf n_iter natmu opt_det opt_stoch α σ_init)
    if !isdir(target_dir)
        return nothing, nothing, nothing, nothing
    end
    res = collect_results(target_dir)
    thinning = 10
    res = ([vcat(vec.(x)...) for x in res.parameters[1:thinning:end]])
    SWA_sqrt_diag = Diagonal(StatsBase.std(res))
    SWA = mean(res[end-K+1:end])
    SWA_D = reduce(hcat, res[end-K+1:end] .- Ref(SWA))
    preds = []
    @showprogress for i in 1:n_MC
        θ = SWA + SWA_sqrt_diag / sqrt(2f0) * randn(Float32, n_θ) + SWA_D / sqrt(2f0 * (K - 1)) * randn(Float32, K)
        pred = nn_forward(X_test, θ)
        push!(preds, cpu(Flux.softmax(pred)))
    end
    mean_preds = mean(preds)
    nll = Flux.Losses.crossentropy(mean_preds, y_test)
    acc = mean(Flux.onecold(mean_preds) .== Flux.onecold(y_test))
    conf, conf_acc = conf_and_acc(mean_preds)
    return nll, acc, conf, conf_acc
end

function extract_info(::Union{Val{:elrgvi},Val{:slang}}, alg_dir, mf, exp_params)
    return nothing, nothing, nothing, nothing
end