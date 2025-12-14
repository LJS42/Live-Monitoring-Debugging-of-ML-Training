module LMD4MLTraining

include("cockpit/session.jl")
include("cockpit/quantities.jl")
include("cockpit/instruments.jl")
include("cockpit/utils.jl")

include("visualization/plots.jl")
include("visualization/dashboard.jl")

include("backends/flux.jl")

export
    Session,
    LossQuantity,
    GradNormQuantity,
    show_cockpit,
    train_with_cockpit

end
