## VI
using Random
using AdvancedVI; const AVI = AdvancedVI

using ProtoStructs  # TODO use real structs


@proto VIExperiment <: ExperimentSpecification
# struct VIExperiment
#     study_name::String
#     model
#     algo::Algo
#     q::Function
#     optimizer::optimizer
#     random_seed::Int
#     # additional dict to be stored
#     additional_dict::Dict
# end


@proto VIResults
# struct VIResults
# end

function get_experiment_results(exp::VIExperiment)
    Random.seed!(exp.random_seed)
    # TODO should vi be called vi!, it appers to change q!!! (or it should make a copy of q)
    AVI.vi(exp.model, exp.algo, exp.q(), exp.optimizer)
    # TODO Store more results? Add callback?
    results = copy(exp.additional_dict)
    results["q"] = q
    return Dict(
        "it00000" => results
    )
end


# Example usage

# some code for results analysis
include(srcdir("models/bnn.jl"))

bnn_example = logπ_base

function create_bnn_experiment()
    VIExperiment(
        study_name = 'test'
        model = bnn_example
        algo = AVI.PFlowVI(max_iters, false, true)
        q = () -> SamplesMvNormal(copy(θ_t))
        optimizer = ADAGrad(0.1)
        random_seed = 0
        additional_dict = Dict(
            "model" => "bnn",
        )
    ]
end


create_bnn_experiment()
#
# for model in models
#     for algo in algos
#         run_experiment(model, algos, additional_dict)
#     end
# end
