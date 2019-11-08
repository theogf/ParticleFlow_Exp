using ForwardDiff
using GradDescent
free_energy(phi,x) = -0.5*logdet(cov(x,dims=2)) + expec(phi,x)
grad_hyper_free_energy(l,x_t) = ForwardDiff.gradient(x->free_energy(phi,x_t,x),l)
expec(phi,x,invK) = mean(phi.(eachcol(x),[invK]))
# (phi,x) = -free_energy(phi,x)
f(x,invK) = -0.5*ForwardDiff.gradient(x->phi(x,invK),x)
f!(g,x) = g.=f(x)
sum_f(f) = mean(f,dims=2)
add_var_f(x,f,m) = (0.5I + sum_var_f(x,f,m))*(x.-m)
sum_var_f(x,f,m) = mean(eachcol(f).*transpose.(eachcol(x.-m)))
mu_and_sig(x) = vec(mean(x,dims=2)),cov(x,dims=2)

function move_particles(x_t,θ,f,T=10,opt_x=[Momentum(η=0.001),Momentum(η=0.001)],opt_θ=Adam(α=0.001);cb=nothing,Xt=nothing)
    invK = inv(exp(θ[1])*kernelmatrix(SqExponentialKernel(exp(θ[2])),reshape(X,:,1),obsdim=1)+1e-5I)
    L = free_energy(phi,x_t,invK)
    L_new = Inf
    x_new = copy(x_t)
    @progress for t in 1:T
        m = mean(x_t,dims=2)
        f_x = mapslices(x->f(x,invK),x_t,dims=1)
        Δx = update(opt_x[1],sum_f(f_x))
        var_x = update(opt_x[2],add_var_f(x_t,f_x,m))
        # @info Δx, var_x[:,1]
        x_new .= x_t .+ Δx .+ var_x
        L_new = free_energy(phi,x_new,invK)
        @show L,L_new
        α = 0.5
        while L_new > L
            # @info α
            x_new .= x_t .+ α*Δx .+ α*var_x
            L_new = free_energy(phi,x_new,invK)
            α *= 0.5
            if α < 1e-10
                @error "α too small"
                continue
            end
        end
        @info α
        x_t .= x_new
        grad_θ = grad_hyper_free_energy(θ,x_t)
        # θ .+= update(opt_θ,grad_θ)
        @info θ
        invK .= inv(exp(θ[1])*(kernelmatrix(SqExponentialKernel(exp(θ[2])),reshape(X,:,1),obsdim=1)+1e-5I))
        free_energy(phi,x_t,invK)
        L = L_new
        if !isnothing(Xt)
            push!(X_t,copy(x_t))
        end
        if !isnothing(cb)
            cb(x_t,invK,t)
        end
    end
end
