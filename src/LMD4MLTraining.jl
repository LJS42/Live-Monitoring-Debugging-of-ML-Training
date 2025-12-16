module LMD4MLTraining

using Flux
using WGLMakie
using Bonito
using Makie
using Dates
using Statistics
using Base.Threads
using Logging

# Core cockpit and quantities
include("quantities/quantity.jl")
include("quantities/loss.jl")
include("quantities/gradnorm.jl")
include("learner.jl")
include("instruments/renderer.jl")

export
    Learner,
    LossQuantity,
    GradNormQuantity,
    Train!
end
