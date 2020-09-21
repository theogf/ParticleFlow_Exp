# First callback part is for returning a MVHistory object, for light problems
using BSON


cb_tic(h, i::Int) = push!(h, :t_tic, Float64(time_ns()) / 1e9)
cb_toc(h, i::Int) = push!(h, :t_toc, Float64(time_ns()) / 1e9)

# Return a callback function using the correct history
function wrap_cb(; cb_hp = nothing, cb_val = nothing)
    return function base_cb(h::MVHistory)
        return function (i, q, hp)
            if iseverylog10(i)
                cb_tic(h, i)
                if !isnothing(hp) && !isnothing(cb_hp)
                    cb_hp(h, i, hp)
                end
                cb_var(h, i, q)
                isnothing(cb_val) ? nothing : cb_val(h, i, q, hp)
                cb_toc(h, i)
            end
        end
    end
end

# Callback function on hyperparameters
function cb_hp_svgp(h, i::Int, hp)
    push!(h, :Z, i, hp[1:end-2])
    push!(h, :σ_kernel, i, hp[end-1])
    push!(h, :l_kernel, i, hp[end])
end

# Wrapper for transformed distributions
cb_var(h, i::Int, q::TransformedDistribution) = cb_var(h, i, q.dist)

# Store mean and covariance
function cb_var(h, i::Int, q::Union{AVI.AbstractSamplesMvNormal,AVI.SteinDistribution})
    push!(h, :mu, i, Vector(mean(q)))
    push!(h, :sig, i, Vector(cov(q)[:]))
end

function cb_var(h, i::Int, q::AVI.MFSamplesMvNormal)
    push!(h, :mu, i, Vector(mean(q)))
    push!(h, :sig, i, Vector(vcat(vec.(blocks(cov(q)))...)))
    push!(h, :indices, i, Vector(q.id))
end

function cb_var(h, i::Int, q::TuringDenseMvNormal)
    push!(h, :mu, i, q.m)
    push!(h, :sig, i, Matrix(q.C)[:])
end

# Second callback is for NN which require to write on file to avoid allocation

# Return a callback function using the correct history
function wrap_heavy_cb(; cb_hp = nothing, cb_val = nothing, path = nothing)
    return function base_heavy_cb(h::MVHistory)
        return function (i, q, hp)
            if iseverylog10(i)
                cb_tic(h, i)
                if !isnothing(hp) && !isnothing(cb_hp)
                    cb_hp(h, i, hp)
                end
                isnothing(path) ? nothing : cb_heavy_var(h, i, q, path)
                isnothing(cb_val) ? nothing : cb_val(h, i, q, hp)
                cb_toc(h, i)
            end
        end
    end
end



# Wrapper for transformed distributions
cb_heavy_var(h, i::Int, q::TransformedDistribution, path::String) = cb_heavy_var(h, i, q.dist, path)

# Store mean and covariance
function cb_heavy_var(h, i::Int, q, path::String)
    @info "Saving model at iteration $i in $path"
    isdir(path) ? nothing : mkdir(path)
    new_path = joinpath(path, string("model_iter_", i, ".bson"))
    q = cpu(q)
    BSON.@save new_path q
end
