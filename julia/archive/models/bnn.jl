using Flux

mutable struct BNNModel <: AbstractModel
    likelihood::Function
    outputfunction::Function
    nLayers::Int
    nNeurons::Vector{Int}
    nInputs::Int
    nParams::Int
    prior::Function
    X::AbstractArray
    y::Vector
    m::Vector
    C::Matrix
end

function BNNModel(likelihood,output,nNeurons,X,y,prior)
    nParams = _nParams(nNeurons,X)
    BNNModel(likelihood,output,length(nNeurons),nNeurons,size(X,2),nParams,prior,X,y,zeros(nParams),zeros(nParams,nParams))
end

function unpack(bnn::BNNModel,parameters)
    @assert length(parameters) == bnn.nParams
    Ws = []
    bs = []
    @show nNeurons = vcat(bnn.nInputs,bnn.nNeurons)
    parser = 1
    for i in 1:bnn.nLayers
        nW = nNeurons[i]*nNeurons[i+1]
        push!(Ws,reshape(parameters[parser:parser+nW-1],nNeurons[i+1],nNeurons[i]))
        parser += nW
        push!(bs,reshape(parameters[parser:parser+nNeurons[i+1]-1],nNeurons[i+1]))
        parser += nNeurons[i+1]
    end
    return Chain(Dense.(Ws[1:end-1],bs[1:end-1])...,Dense(Ws[end],bs[end],bnn.outputfunction))
end

_nParams(nNeurons,X) = sum((vcat(size(X,1),nNeurons[1:end-1]).+1).*nNeurons)

# X = (rand(200,2).-0.5)*2
# y = (Int64.(sign.(X[:,1].*X[:,2])).+1).÷2
# # using Plots
# # scatter(eachcol(X)...,color=y,lab="")
# model = BNNModel((ŷ,y)->Flux.crossentropy(ŷ,y),x->Flux.σ(x),[2,3,1],X,y,x->pdf(MvNormal(zeros(_nW([2,3,1],X)))))
#
#
# chain = unpack(model,rand(model.nParams))
# M = 100
# x_t =
# chain(X[1,:])
# model.nInputs
