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
    d1 = MvNormal(zeros(2),I)
    p_x = pdf.(Ref(d1),collect.(X))
    scene = contour(xrange,yrange,p_x,levels=10,fillrange=true,linewidth=0.0)
    d2 = MvNormal([-5,5],I)
    p_x2 = pdf.(Ref(d2),collect.(X))
    levels = inv(det(cov(d2))*2π)*exp.(-0.5(5:-1:1)) # Return levels at 1:5 sigmas
    scene = contour!(scene,xrange,yrange,p_x2,color=:white,levels=levels)
    # AbstractPlotting.inline!(true)

    scene = contour(xrange,xrange,p_x_target,levels=100,fillrange=true,linewidth=0.0)
    scene = contour!(xrange,xrange,p_X,color=:white,levels=levels)
    scene = scatter!(x_p1,x_p2,color=:red,markersize=0.3)
    scene = title(scene,titlestring)
    return scene, x_p, tstring
end
