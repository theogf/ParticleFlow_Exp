using DrWatson
@quickactivate
using Pkg; Pkg.update()

using Distributed, CUDA
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
    :n_epoch => 100,
    :batchsize => 128,
    :model => "LeNet",
    :dataset => "MNIST",
    :use_gpu => true,
    :start_layer => [1, 7, 8, 9],
    :seed => 42,
    :n_period => 10,
    :η => 1f-3, # 0.001 in Float32
    :α => 0.01,
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
