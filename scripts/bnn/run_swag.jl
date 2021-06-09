using DrWatson
@quickactivate
using Pkg; Pkg.update()

using CUDA
# using Distributed
# if length(workers()) != length(devices()) || first(workers()) == 1
#     addprocs(length(devices()))
# end
# @everywhere using DrWatson
# @everywhere @quickactivate
# @everywhere using CUDA

# Assign one GPU per worker
# asyncmap((zip(workers(), devices()))) do (p, d)
#     remotecall_wait(p) do
#         @info "Worker $p uses $d"
#         context()
#         device!(d)
#     end
# end

include(srcdir("bnn", "swag.jl"))
# @everywhere include(srcdir("bnn", "swag.jl"))

exp_ps = Dict(
    :n_epoch => 50,
    :batchsize => 128,
    :n_hidden => [100, 200, 400, 800],
    :activation => [:tanh, :relu],
    :model => "BNN",
    :dataset => "MNIST",
    :use_gpu => true,
    :seed => 42,
    :n_period => 10,
    :eta => 1f-2,#[1f-1, 5f-2, 1f-2], # 0.001 in Float32
    :α => [0.01, 0.05, 0.1, 1.0, 5.0, 10, 50, 100],
)

ps = dict_list(exp_ps)
@info "Will now run $(dict_list_count(exp_ps)) simulations"

# pmap(run_SWAG, ps)
# run_SWAG(ps[4])

for (i, p) in enumerate(ps)
    @info "Running dict $(i)/$(length(ps)) : $(savename(p))"
    run_SWAG(p)
end

# run_SWAG(ps[3])
