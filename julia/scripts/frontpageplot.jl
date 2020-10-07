using Makie, ColorSchemes
using AbstractPlotting.MakieLayout
using AdvancedVI
using Distributions, LinearAlgebra
using ForwardDiff: gradient
using Flux.Optimise
using Random: seed!
seed!(1234)
m1 = [0, 0]
m2 = [10.0, 0.0]
K = 10
x0 = randn(2, K)
d1 = MvNormal(m1, 1.5 * [1 0; 0 1])
d2 = MvNormal(m2, [1 -1.0; -1.0 3])
colors = ColorSchemes.seaborn_colorblind
cmap = ColorSchemes.thermal
x1 = cholesky(cov(d1)).L * x0
function is_std(d, nσ)
    θ = range(0, 2π, length = 100)
    return mean(d) .+ sqrt(nσ) * cholesky(cov(d)).L * [cos.(θ) sin.(θ)]'
end

## Compute gradients
xs = []; ds = []; gs= []; ms =[]
push!(xs, x0); push!(ds, d1); push!(ms, mean(x0))
η0 = Optimise.Optimiser(ClipNorm(1.0), Descent(1.0))
# η0 = Descent(1.0)
η1 = Optimise.Optimiser(ClipNorm(1.0), Descent(1.0))
η1 = Descent(1.0)
for i in 1:10
    g = mapslices(xs[i]; dims=1) do x
        gradient(x) do y
            -logpdf(d2, y)
        end
    end
    mean_g = -Optimise.apply!(η0, [], vec(mean(g, dims= 2)))
    m = mean(xs[i], dims = 2)
    xgrads = - Optimise.apply!(η0, [], (mean(eachcol(g).*transpose.(eachcol(xs[i] .- m)))- I) * (xs[i] .- m))
    xgrads .+= mean_g
    xnext = xs[i] + Optimise.apply!(η1, [], xgrads)
    push!(xs, xnext); push!(ds, SamplesMvNormal(xnext))
    push!(gs, xgrads)
end
xfinal = cholesky(cov(d2)).L * x0  .+ m2
valint = 6
dint = ds[valint]; xint= xs[valint]; gint = gs[valint]

## Plotting
cmap = range(colors[1], colors[3], length = 3)
colormarker = colors[2]
scene, layout = layoutscene(resolution = (1200, 600))
ax = layout[1,1] = LAxis(scene, aspect = DataAspect())
line_color = Dict(d1=>cmap[1], d2=>cmap[end], dint => cmap[2])
line_style = Dict(d1=>:solid, d2=>:solid, d3 => :dash, dint=> :dash)
for d in [d1, d2, dint]
    for i in 1:3
        lines!(ax, eachrow(is_std(d, i))..., color = line_color[d], linewidth = 5.0, linestyle = line_style[d])
    end
end
thinning = 2
N = length(ds[1:thinning:end])

# for (j,d) in enumerate(ds[1:thinning:end])
#     for i in 1:3
#         lines!(ax, eachrow(is_std(d, i))..., color = cmap[j], linewidth = 5.0, linestyle = line_style[d1])
#     end
# end
# for (j,x) in enumerate(xs[1:thinning:end])
#     scatter!(ax, eachrow(x)..., color = cmap[j], strokewidth=0.0, markersize = 15.0)
# end
arrows!(ax, x1[1,:], x1[2,:], gs[1][1,:], gs[1][2,:], linewidth = 3.0, arrowsize = 0.13, linecolor = colormarker, arrowcolor = colormarker)
scatter!(ax, eachrow(x1)..., color = colormarker, strokewidth=0.0, markersize = 15.0)
arrows!(ax, xint[1,:], xint[2,:], gint[1,:], gint[2,:], linewidth = 3.0, arrowsize = 0.13, linecolor = colormarker, arrowcolor = colormarker)
scatter!(ax, eachrow(xint)..., color = colormarker, strokewidth=0.0, markersize = 15.0)
scatter!(ax, eachrow(xfinal)..., color = colormarker, strokewidth=0.0, markersize = 15.0)
arrows!(ax, [first(mean(d1))], [last(mean(d1))],  [first(mean_g)], [last(mean_g)], linewidth = 7.0, arrowsize = 0.4, linecolor = cmap[1], arrowcolor = cmap[1])
arrows!(ax, [first(mean(dint))], [last(mean(dint))],  [first(mean(gint, dims =2))], [last(mean(gint, dims = 2))], linewidth = 7.0, arrowsize = 0.4, linecolor = cmap[2], arrowcolor = cmap[2])
ts = 0.8
text!(ax, "q₀(x)", position = Point2f0(m1.-[0,-4.0]), textsize = ts, align =(:center, :center))
text!(ax, "p(x)", position = Point2f0(m2.-[0.0,-4.0]), textsize = ts, align =(:center, :center))

hidedecorations!(ax)
hidespines!(ax)
save(joinpath("/home", "theo", "experiments", "ParticleFlow", "julia", "plots", "frontpage.png"), scene)
cp(joinpath("/home", "theo", "experiments", "ParticleFlow", "julia", "plots", "frontpage.png"), joinpath("/home", "theo", "Tex Projects", "GaussianParticleFlow", "figures", "frontpage.png"), force =true)
scene