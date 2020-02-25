using Makie, Distributions, Colors

function set_plotting_scene_2D(x,xrange,yrange,p_x_target)
    tstring = Node("t=0")
    x_p = Node(x)
    m,C = m_and_C(x)
    X = Iterators.product(xrange,yrange)
    dist_x = lift(x->MvNormal(m_and_C(x)...),x_p)
    p_X = lift(x->pdf.(Ref(x),collect.(X)),dist_x)
    x_p1  = lift(x->x[1,:],x_p)
    x_p2  = lift(x->x[2,:],x_p)
    levels = lift(x->Float32.(normalizer(x)*exp.(-0.5(5:-1:1))),dist_x)
    d2 = MvNormal(m,I)
    p_x2 = pdf.(Ref(d2),collect.(X))
    levels = inv(det(cov(d2))*2π)*exp.(-0.5(5:-1:1)) # Return levels at 1:5 sigmas
    # scene = contour!(scene,xrange,yrange,p_x2,color=:white,levels=levels)
    # AbstractPlotting.inline!(true)

    scene = contour(xrange,xrange,p_x_target,levels=100,fillrange=true,linewidth=0.0)
    # scene = contour!(xrange,xrange,p_X,color=:white,levels=levels)
    scene = scatter!(x_p1,x_p2,color=:red,markersize=0.2)
    scene = title(scene,tstring)
    return scene, x_p, tstring
end


function set_plotting_scene_GP(x,p,xrange,y,xpred,μgp,siggp,∇f)
    xsort= sortperm(xrange)
    scene = Makie.scatter(sort(xrange),y[xsort],markersize=0.1)
    plot!(xpred,μgp)
    plot!(xpred,μgp.+sqrt.(max.(0.0,siggp)),linestyle=:dash)
    plot!(xpred,μgp.-sqrt.(max.(0.0,siggp)),linestyle=:dash)
    x_p = Node(x)
    m_and_sig = lift(x->predic_f(p,x,reshape(xpred,:,1)),x_p)
    mf = lift(x->x[1],m_and_sig)
    sigf = lift(x->sqrt.(max.(0.0,x[2])),m_and_sig)
    mfplus = lift(x->x[1].+sqrt.(max.(0.0,x[2])),m_and_sig)
    mfminus = lift(x->x[1].-sqrt.(max.(0.0,x[2])),m_and_sig)
    Makie.plot!(xpred,mf,color=:blue,linewidth=2.0)
    Makie.fill_between!(xpred,mfminus,mfplus,color=RGBA(colorant"green",0.3))
    ∇f_p = Node(∇f)
    arrows!(xrange,zero(xrange),zero(xrange),∇f_p,arrowsize=0.1)
    # particles = [lift(x->predic_f(p,x[:,i],reshape(xpred,:,1)),x_p) for i in 1:size(x,2)]
    # plot!.([xpred],particles,color=RGBA(0.0,0.0,0.0,0.1))
    return scene, x_p,∇f_p
end
