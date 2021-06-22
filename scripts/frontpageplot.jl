using Makie, ColorSchemes
using AbstractPlotting.MakieLayout
using AdvancedVI
using Distributions, LinearAlgebra
using ForwardDiff: gradient
using Flux.Optimise
using Random: seed!
using Animations
seed!(15)
m1 = [0, 0]
m2 = [10.0, 0.0]
K = 6
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
η0 = Descent(1.0)
# η1 = Optimise.Optimiser(ClipNorm(1.0), Descent(1.0))
η1 = Descent(0.2)
T = 200
for i in 1:T
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


## Plotting
valint = 3#div(T, 2)
dstart = ds[1]; xstart = xs[1]; gstart = gs[1]
dint = ds[valint]; xint= xs[valint]; gint = gs[valint]
xfinal = xs[end]
cmap = range(colors[1], colors[2], length = 3)
λ = 0.1
cweights = [1 - exp(-λ * w) * exp(λ) for w in range(0, 1, length = T+1)]
canim = Animation(
    1, colors[1],
    polyout(25),
    T+1, colors[2]
)
cmap[2] = at(canim, valint)
fullcmap = at.(canim, 1:T+1)
colormarker = colors[2]
scene, layout = layoutscene(resolution = (1200, 600))
ax = layout[1,1] = LAxis(scene, aspect = DataAspect())
line_color = Dict(d1=>cmap[1], d2=>cmap[end], dint => cmap[2])
line_style = Dict(d1=>:solid, d2=>:solid, dint=> :solid)
arrows_points = [1, 3, 5, 30]
for i in arrows_points
    arrows!(ax, xs[i][1,:], xs[i][2,:], xs[i+1][1,:] - xs[i][1,:], xs[i+1][2,:] - xs[i][2, :], color = at(canim, i), arrowcolor = at(canim, i))
end
for i in 1:K
    xline = reduce(hcat, x[:, i] for x in xs)
    lines!(ax, eachrow(xline)..., color = fullcmap, linewidth = 5.0)
end
for d in [d1, d2, dint]
    for i in 1:3
        lines!(ax, eachrow(is_std(d, i))..., color = line_color[d], linewidth = 5.0, linestyle = line_style[d])
    end
end
thinning = 2
N = length(ds[1:thinning:end])


gscale = 1.5
swidth = 1.5
scatter!(ax, eachrow(xstart)..., color = cmap[1], strokewidth=swidth, markersize = 15.0, strokecolor = :white)
scatter!(ax, eachrow(xint)..., color = cmap[2], strokewidth=swidth, markersize = 15.0, strokecolor = :white)
scatter!(ax, eachrow(xfinal)..., color = cmap[3], strokewidth=swidth, markersize = 15.0, strokecolor = :white)
arrows!(ax, [first(mean(dstart))], [last(mean(dstart))],  [first(mean(gstart, dims = 2))] * gscale, [last(mean(gstart, dims = 2))] * gscale, linewidth = 7.0, arrowsize = 0.4, linecolor = cmap[1], arrowcolor = cmap[1])
arrows!(ax, [first(mean(dint))], [last(mean(dint))],  [first(mean(gint, dims =2))] * gscale, [last(mean(gint, dims = 2))] * gscale, linewidth = 7.0, arrowsize = 0.4, linecolor = cmap[2], arrowcolor = cmap[2])
ts = 0.8
text!(ax, "q⁰(x)", position = Point2f0(m1.-[0,-3.0]), textsize = ts, align =(:center, :center))
text!(ax, "qᵗ(x)", position = Point2f0(vec(mean(xint, dims = 2)).-[0.0,-2.7]), textsize = ts, align =(:center, :center))
text!(ax, "p(x)", position = Point2f0(m2.-[-1.5,-3.5]), textsize = ts, align =(:center, :center))
hidedecorations!(ax)
hidespines!(ax)
save(joinpath(@__DIR__, "..", "plots", "frontpage.png"), scene)