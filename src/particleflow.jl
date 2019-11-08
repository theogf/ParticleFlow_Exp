using ForwardDiff
using Distributions, LinearAlgebra


# Two modes
phi(x) = 0.5*(x[1]^2*x[2]^2+x[1]^2+x[2]^2-8x[1]-8x[2])
# Normal Gaussian
μ = zeros(2); Σ = diagm(ones(2))
# phi(x) = 0.5*dot(x.-μ,inv(Σ)*(x.-μ))


xrange = range(-2.5,7.5,length=100)
X = Iterators.product(xrange,xrange)
p_x = zeros(size(X))
for (i,x) in enumerate(X)
    p_x[i] = exp(-phi(x))
end

p_x
using Plots; pyplot()
# surface(xrange,xrange,p_x,levels=100,lw=0.0)
##
M = 20
eta_1 = 0.1
eta_2 = 0.1
mu0 = zeros(2)
C = diagm(ones(2))
x_init = rand(MvNormal(mu0,C),M)
x_t = copy(x_init)
X_t = []
m = mean(x_t,dims=2)
f_t = mapslices(f,x_t,dims=1)
# sum_f(f_t)
function cb(x_t,t)
    if t%10 ==0
        L = F(phi,x_t)
        p = contourf(xrange,xrange,p_x,levels=100,lw=0.0,title="t=$t, L=$L",cbar=false)
        m,S = mu_and_sig(x_t)
        d = MvNormal(m,S)
        p_q = zeros(size(X))
        for (i,x) in enumerate(X)
            p_q[i] = pdf(d,collect(x))
        end
        contour!(xrange,xrange,p_q',lab="",color=:white)
        scatter!(eachrow(x_t)...,lab="")
        frame(a)
    end
end
f_x = mapslices(f,x_t,dims=1)
x_t .+= eta_1*sum_f(f_x) .+ eta_2*add_var_f(x_t,f_x,m)
a = Animation()
move_particles(x_t,f,1000,cb)
x_t
gif(a) |> display

mu_and_sig.(X_t)
mu_and_sig(x_t)
