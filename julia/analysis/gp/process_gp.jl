using DrWatson
@quickactivate
include(projectdir("analysis", "post_process.jl"))
include(srcdir("utils", "gp.jl"))
include(srcdir("utils", "tools.jl"))
using AxisArrays, MCMCChains, PDMats, KernelFunctions

dataset = "ionosphere"
(X_train, y_train), (X_test, y_test) = load_gp_data(dataset)
ρ = initial_lengthscale(X_train)
k = KernelFunctions.transform(SqExponentialKernel(), 1 / ρ)
K = kernelpdmat(k, X_train, obsdim = 1)
Ktt = kerneldiagmatrix(k, X_test, obsdim = 1)
Ktx = kernelmatrix(k, X_test, X_train, obsdim = 1)
function pred_f(f)
    Ktx * inv(K) * f
end


all_res = collect_results(datadir("results", "gp", dataset))
gpf_res = @linq all_res |> where(:n_iters .=== 1000)
vi_res = @linq all_res |> where(:vi .=== true)
vi_res[:μ_f][1]
q = gpf_res[:q][1]
mean_and_var(pred_f, q)
