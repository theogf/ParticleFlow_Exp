using .Turing

function vi(model::Turing.Model, alg::SVGD, n_particles::Int ; optimizer = TruncatedADAGrad(), callback = nothing)
    logπ = Turing.Variational.make_logjoint(model)
    nVars = sum(length(v.ranges) for v in values(Turing.VarInfo(model).metadata))

    q = transformed(EmpiricalDistribution(randn(nVars, n_particles)), bijector(model))
    vi(logπ, alg, q; optimizer = optimizer, callback = callback)
end

function vi(model::Turing.Model, alg::GaussPFlow, n_particles::Int ; optimizer = TruncatedADAGrad(), callback = nothing)
    logπ = Turing.Variational.make_logjoint(model)
    nVars = sum(length(v.ranges) for v in values(Turing.VarInfo(model).metadata))
    
    q = transformed(SamplesMvNormal(randn(n_particles, nVars)),bijector(model))
    vi(logπ, alg, q; optimizer = optimizer, callback = callback)
end
