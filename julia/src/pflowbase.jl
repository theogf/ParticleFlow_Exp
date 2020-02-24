using ForwardDiff, LinearAlgebra
using Zygote
using KernelFunctions
using Distributions
using PDMats

abstract type AbstractModel end


include("models/generalmodel.jl")
include("models/gpmodel.jl")
include("models/normflow.jl")
include("models/bnn.jl")

Flux.@functor ScaleTransform
Flux.@functor ARDTransform
Flux.@functor SqExponentialKernel

## General Helper  ###
@inline sum_f(f) = mean(f,dims=2)
@inline add_var_f(x,f,m) = (0.5I + sum_var_f(x,f,m))*(x.-m)
@inline sum_var_f(x,f,m) = mean(eachcol(f).*transpose.(eachcol(x.-m)))
@inline m_and_C(x) = vec(mean(x,dims=2)),cov(x,dims=2)+1e-5I
@inline _C(x) = cov(x,dims=2)
@inline m_and_diagC(x) = vec(mean(x,dims=2)),diag(cov(x,dims=2))
no_prior(x,p) = zero(eltype(x))
base_kernel(k::Kernel) = eval(nameof(typeof(k)))
normalizer(d::MvNormal) = inv(det(cov(d))*2π)^(0.5*length(d))


## Main function
function move_particles(x,p,opt_x;cb=nothing,Xt=nothing,epsilon=1e-3,precond_b=true,precond_A=true)
    # L = free_energy(x,p)
    # L_new = Inf
    x_new = copy(x)
    p.m = vec(mean(x,dims=2))
    f_x = mapslices(x->_f(x,p),x,dims=1)
    global ∇f1 = if precond_b
            vec(mean(f_x,dims=2))
        else
            _C(x)*vec(mean(f_x,dims=2))
        end
    c_x = x.-p.m
    ψ = mean(eachcol(f_x).*transpose.(eachcol(c_x)))
    A = ψ+0.5I
    global ∇f2 = if precond_A
            2*tr(A'*A)/(tr(A^2)+tr(A'*inv(_C(x))*A*_C(x)))*A*c_x
        else
            A*c_x
        end
    Δ1 = Flux.Optimise.apply!(opt_x[1],p.m,∇f1)
    Δ2 = Flux.Optimise.apply!(opt_x[2],x,∇f2)
    @. x_new = x + Δ1 + Δ2
    # L_new = free_energy(x_new,p)
    α = 0.5
    # while L_new > L+epsilon
    #     x_new .= x .+ α*Δ1 .+ α*Δ2
    #     L_new = free_energy(x_new,p)
    #     α *= 0.1
    #     @show α
    #
    #     if α < 1e-10
    #         @error "α too small, skipping step!"
    #         α = 0.0
    #         @. x_new = x + α*Δ1 + α*Δ2
    #         break
    #     end
    # end
    @. x = x_new
    update_params!(p,x,p.opt)
    # L = L_new
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
