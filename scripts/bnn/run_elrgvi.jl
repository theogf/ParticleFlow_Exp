using DrWatson
@quickactivate

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

include(srcdir("bnn", "elrgvi.jl"))
# @everywhere include(srcdir("bnn", "swag.jl"))
GC.gc(true)
CUDA.reclaim()
exp_ps = Dict(
    :n_iter => 5001,
    :batchsize => 128,
    :n_hidden => 200,#[100, 200, 400, 800],
    :activation => :tanh, #[:tanh, :relu],
    :L => [2, 5, 10],
    :model => "BNN",
    :dataset => "MNIST",
    :use_gpu => false,
    :seed => 42,
    :α => 1f0,
    :opt => :RMSProp,
    :eta => 1f-2,
)

ps = dict_list(exp_ps)
@info "Will now run $(dict_list_count(exp_ps)) simulations"
CUDA.allowscalar(true)
# ProfileView.@profview run_elgrvi(ps[1])

run_elrgvi(ps[3])


# for (i, p) in enumerate(ps[2:end])
#     @info "Running dict $(i)/$(length(ps)) : $(savename(p))"
#     GC.gc()
#     # CUDA.reclaim()
#     @time run_elrgvi(p)
# end
