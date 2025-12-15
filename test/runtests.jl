using LMD4MLTraining
using Test
using Flux
using Base.Threads
using WGLMakie
using Makie

# GradNormQuantity
@testset "GradNormQuantity" begin
    q = GradNormQuantity()
    grads = (a = rand(3), b = rand(2,2))
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

# Learner + train_loop! + Train!
@testset "Learner training" begin
    model = Chain(Dense(2,2))
    data = [(rand(Float32,2), rand(Float32,2))]
    loss_fn(ŷ, y) = sum(abs2, ŷ .- y)
    opt = Flux.setup(Adam(3.0f-4), model)
    learner = Learner(model, data, loss_fn, opt, [LossQuantity()])

    ch = Channel{Tuple{Int,Dict{Symbol,Float32}}}(10)
    t1 = @async LMD4MLTraining.train_loop!(learner, 1, ch)
    wait(t1)
    @test true

    LMD4MLTraining.Train!(learner, 1, false)
    @test true
end