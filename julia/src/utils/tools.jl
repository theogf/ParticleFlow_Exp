# Return true for i = 1,2,3,..10,20,30,..,100,200 etc
iseverylog10(iter) = iter % (10^(floor(Int64, log10(iter + 1)))) == 0


# Return the mean and the variance over all particles
function mean_and_var(f::Function, q::AVI.AbstractSamplesMvNormal)
    vals = f.(eachcol(q.x))
    return mean(vals), var(vals)
end
