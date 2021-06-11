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

function get_mean_pred(θs)
    preds = []
    @showprogress for θ in eachcol(θs)
        pred = nn_forward(X_test, θ)
        push!(preds, cpu(Flux.softmax(pred)))
    end
    return mean(preds)
end

function treat_mean_preds(m_preds)
    nll = Flux.Losses.crossentropy(m_preds, y_test)
    acc = mean(Flux.onecold(m_preds) .== Flux.onecold(y_test))
    conf, conf_acc = conf_and_acc(m_preds)
    return nll, acc, conf, conf_acc
end

function extract_info(alg::Union{Val{:gpf},Val{:svgd_linear},Val{:svgd_rbf}}, alg_dir, mf, exp_params)
    @unpack L, eta, batchsize, n_iter, natmu, opt_det, opt_stoch, α, σ_init = exp_params
    target_dir = joinpath(alg_dir, @savename L eta batchsize mf n_iter natmu opt_det opt_stoch α σ_init)
    if !isdir(target_dir)
        return nothing, nothing, nothing, nothing
    end
    res = collect_results(target_dir)
    # final_particles = first(res.particles[res.i .== n_iter - 1])
    s = sortperm(res.i)
    accs, nlls, confs, conf_accs = [], [], [], [], []
    for θs in res.particles[s]
        mean_preds = get_mean_pred(θs)
        acc, nll, conf, conf_acc = treat_mean_preds(mean_preds)
        push!(accs, acc); push!(nlls, nll); push!(confs, conf); push!(conf_accs, conf_acc)
    end
    return accs, nlls, confs, conf_accs
end

function extract_info(::Union{Val{:gf},Val{:dsvi}}, alg_dir, mf, exp_params)
    @unpack L, batchsize, eta, n_iter, natmu, opt_det, opt_stoch, α, σ_init = exp_params
    target_dir = joinpath(alg_dir, @savename L batchsize mf n_iter natmu opt_det opt_stoch eta α σ_init)
    if !isdir(target_dir)
        return nothing, nothing, nothing, nothing
    end
    res = collect_results(target_dir)
    s = sortperm(res.i)
    accs, nlls, confs, conf_accs = [], [], [], [], []
    for q in res.q[s]
        mean_preds = get_mean_pred(rand(q, n_MC))
        acc, nll, conf, conf_acc = treat_mean_preds(mean_preds)
        push!(accs, acc); push!(nlls, nll); push!(confs, conf); push!(conf_accs, conf_acc)
    end
    return accs, nlls, confs, conf_accs
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
    q_SWA = AVI.FCSMvNormal(SWA, SWA_D / sqrt(2f0), SWA_sqrt_diag / (sqrt(2f0 * (K - 1))))
    mean_preds = get_mean_pred(rand(q_SWA, n_MC))
    return treat_mean_preds(mean_preds)
end

function extract_info(::Union{Val{:elrgvi},Val{:slang}}, alg_dir, mf, exp_params)
    return nothing, nothing, nothing, nothing
end