using ForwardDiff, LinearAlgebra
using GradDescent, Zygote
using KernelFunctions
using Distributions
using PDMats

abstract type AbstractModel end


include("models/generalmodel.jl")
include("models/gpmodel.jl")
include("models/normflow.jl")
## General Helper  ###
@inline sum_f(f) = mean(f,dims=2)
@inline add_var_f(x,f,m) = (0.5I + sum_var_f(x,f,m))*(x.-m)
@inline sum_var_f(x,f,m) = mean(eachcol(f).*transpose.(eachcol(x.-m)))
@inline m_and_C(x) = vec(mean(x,dims=2)),cov(x,dims=2)+1e-5I
@inline m_and_diagC(x) = vec(mean(x,dims=2)),diag(cov(x,dims=2))
no_prior(x,p) = zero(eltype(x))
base_kernel(k::Kernel) = eval(nameof(typeof(k)))
normalizer(d::MvNormal) = inv(det(cov(d))*2π)^(0.5*length(d))


## Main function
function move_particles(x,p,opt_x;cb=nothing,Xt=nothing,epsilon=1e-3)
    L = free_energy(x,p)
    L_new = Inf
    x_new = copy(x)
    m = vec(mean(x,dims=2))
    f_x = mapslices(x->_f(x,p),x,dims=1)
    ∇f1 = vec(mean(f_x,dims=2))
    c_x = x.-m
    ψ = mean(eachcol(f_x).*transpose.(eachcol(c_x)))
    ∇f2 = (ψ+0.5I)*c_x
    Δ1 = update(opt_x[1],∇f1)
    Δ2 = update(opt_x[2],∇f2)
    @. x_new = x + Δ1 + Δ2
    L_new = free_energy(x_new,p)
    α = 0.5
    while L_new > L+epsilon
        x_new .= x .+ α*Δ1 .+ α*Δ2
        L_new = free_energy(x_new,p)
        α *= 0.1
        if α < 1e-10
            @error "α too small, skipping step!"
            α = 0.0
            @. x_new = x + α*Δ1 + α*Δ2
            break
        end
    end
    x .= x_new
    update_params!(p,x,p.opt)
    L = L_new
    if !isnothing(Xt)
        push!(X_t,copy(x_t))
    end
    if !isnothing(cb)
        cb(x,p,iter)
    end
end

function iseverylog10(i)
    i%(10^(floor(Int64,log10(i))))==0
end
