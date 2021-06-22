include(srcdir("algs", "elrgvi.jl"))

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
    @unpack L, batchsize, eta, n_epoch, n_period, α = exp_params
    eta = 1e-6
    α= 1.0
    rho = 0.9
    target_dir = joinpath(alg_dir, @savename batchsize eta rho n_epoch n_period α)
    if !isdir(target_dir)
        return nothing, nothing, nothing, nothing
    end
    res = collect_results(target_dir)
    first_vals = res[res.i .== maximum(res.i), :].parameters[1]
    x = reduce(vcat, vec.(first_vals))
    # return p = get_mean_pred(x)

    thinning = 11
    res = ([reduce(vcat, vec.(x)) for x in res.parameters[1:thinning:end]])
    SWA_sqrt_diag = StatsBase.std(res)
    SWA = mean(res[end-L+1:end])
    SWA_D = reduce(hcat, res[end-L+1:end] .- Ref(SWA))
    q_SWA = AVI.FCSMvNormal(SWA, SWA_D / sqrt(2f0), SWA_sqrt_diag / (sqrt(2f0 * (L - 1))))
    mean_preds = get_mean_pred(rand(q_SWA, n_MC))
    # mean_preds = get_mean_pred(repeat(SWA, 1, 2))#q_SWA, n_MC))
    vals = treat_mean_preds(mean_preds)
    # return SWA
    return [vals[1]], [vals[2]], [vals[3]], [vals[4]]
end

function extract_info(::Val{:slang}, alg_dir, mf, exp_params)
    return nothing, nothing, nothing, nothing
end

function extract_info(::Val{:elrgvi}, alg_dir, mf, exp_params)
    @unpack L, batchsize, eta, n_iter, natmu, opt_det, opt_stoch, α, σ_init = exp_params
    opt = opt_stoch
    target_dir = joinpath(alg_dir, @savename L batchsize n_iter opt eta α)
    if !isdir(target_dir)
        return nothing, nothing, nothing, nothing
    end
    res = collect_results(target_dir)
    s = sortperm(res.i)
    accs, nlls, confs, conf_accs = [], [], [], [], []
    nn = Chain(LowRankGaussianDenseLayer.(m, L)...)
    for ps in res.parameters[s]
        l1, l2, l3 = ps
        μ1, v1, σ1 = l1
        μ2, v2, σ2 = l2
        μ3, v3, σ3 = l3
        nn[1].μ .= μ1; nn[1].v .= v1; nn[1].σ .= σ1
        nn[2].μ .= μ2; nn[2].v .= v2; nn[2].σ .= σ2
        nn[3].μ .= μ3; nn[3].v .= v3; nn[3].σ .= σ3
        mean_preds = get_mean_pred(sample_from_nn(nn, n_MC))
        # mean_preds = get_mean_pred(network_sample(q, n_MC))
        acc, nll, conf, conf_acc = treat_mean_preds(mean_preds)
        push!(accs, acc); push!(nlls, nll); push!(confs, conf); push!(conf_accs, conf_acc)
    end
    return accs, nlls, confs, conf_accs
end