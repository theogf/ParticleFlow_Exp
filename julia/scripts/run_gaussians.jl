using Distributed
nthreads = 5
addprocs(nthreads)
nprocs()
@everywhere using DrWatson
@everywhere quickactivate(@__DIR__)
@everywhere include(srcdir("gaussian", "gaussian_target.jl"))

exp_ps = Dict(
    :n_iters => 5000,
    :n_runs => 10,
    :dim => vcat(1:9),#, 10:10:99, 100:100:500),
    :n_particles => 11,
    :full_cov => false,#[true, false],
    :gpf => true,
    :advi => true,
    :steinvi => true,
    :cond1 => false,
    :cond2 => false,
    :seed => 42,
    :cb_val => nothing,
    :opt => ADAGrad(0.1),
)

ps = dict_list(exp_ps)
run_gaussian_target(ps[1])
pmap(run_gaussian_target, ps)
