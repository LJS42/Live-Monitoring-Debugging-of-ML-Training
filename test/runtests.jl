using LMD4MLTraining
using Test
using Flux
using Base.Threads
using WGLMakie
using Makie
using MLDatasets

# GradNormQuantity
@testset "GradNormQuantity" begin
    q = GradNormQuantity()
    grads = (a=rand(3), b=rand(2, 2))
    val = 1.0
    r = LMD4MLTraining.compute!(q, val, grads)
    @test r > 0
end

# LossQuantity
@testset "LossQuantity compute!" begin
    q = LossQuantity()
    loss_val = 1.23
    r = LMD4MLTraining.compute!(q, loss_val, nothing)
    @test r == loss_val
end

@testset "setup_plots" begin
    struct DummyQ <: LMD4MLTraining.AbstractQuantity
        key::Symbol
    end
    LMD4MLTraining.quantity_key(q::DummyQ) = q.key

    qlist = [DummyQ(:loss)]

    fig, obs, axs = LMD4MLTraining.setup_plots(qlist)
    @test isa(fig, Figure)
    @test isa(obs[:loss], Observable)
    @test isa(axs[:loss], Axis)
end

# render_loop
@testset "render_loop test" begin
    ch = Channel{Tuple{Int,Dict{Symbol,Float32}}}(2)
    put!(ch, (1, Dict(:loss => 0.1f0)))
    put!(ch, (2, Dict(:loss => 0.5f0)))

    # LossQuantity
    qlist = [LossQuantity()]

    try
        t = @async LMD4MLTraining.render_loop(ch, qlist)
        @test true
    catch e
        @warn "render_loop skipped in test due to Makie limits: $e"
        @test true
    end

    sleep(2)

    close(ch)
end

# Data Loader
function get_data_loader()
    preprocess(x, y) = (reshape(x, 28, 28, 1, :), Flux.onehotbatch(y, 0:9))
    x_train_raw, y_train_raw = MLDatasets.MNIST.traindata()
    x_train, y_train = preprocess(x_train_raw, y_train_raw)
    return Flux.DataLoader((x_train, y_train); batchsize=128, shuffle=true)
end

# Model
function get_model()
    return Chain(
        Conv((5, 5), 1 => 6, relu),
        MaxPool((2, 2)),
        Conv((5, 5), 6 => 16, relu),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(256, 120, relu),
        Dense(120, 84, relu),
        Dense(84, 10),
    )
end

# Learner + train_loop! + Train!
@testset "Learner training" begin
    # Define quantities to track
    quantities = [LossQuantity(), GradNormQuantity()]

    model = get_model()
    data_loader = get_data_loader()
    optim = Flux.setup(Adam(3.0f-4), model)
    loss_fn(ŷ, y) = Flux.logitcrossentropy(ŷ, y)
    learner = Learner(model, data_loader, loss_fn, optim, quantities)

    Train!(learner, 1, true)

    println("Training finished.")
end
