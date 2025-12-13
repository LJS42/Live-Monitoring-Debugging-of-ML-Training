using Flux
using Logging

mutable struct Learner
    model::Any
    data_loader::Any
    loss_fn::Function
    optim::Any
    quantities::Vector{<:AbstractQuantity}

    function Learner(model, data_loader, loss_fn, optim, quantities)
        new(model, data_loader, loss_fn, optim, quantities)
    end
end

function Learner(
    model,
    data_loader,
    loss_fn::Function,
    optim
)
    return Learner(model, data_loader, loss_fn, optim, AbstractQuantity[])
end

function Train!(
    learner::Learner,
    epochs::Int,
    with_plots::Bool,
)
    ch = Channel{Tuple{Int,Dict{Symbol,Float32}}}(100)
    train_task = @async train_loop!(
        learner,
        epochs,
        ch,
    )
    if with_plots
        render_task = @async render_loop(ch, learner.quantities)
    end

    wait(train_task)
    wait(render_task)
end

function train_loop!(
    learner::Learner,
    epochs::Int,
    channel::Channel{Tuple{Int,Dict{Symbol,Float32}}},
)
    step_count = 0
    try
        for epoch in 1:epochs
            for (x, y) in learner.data_loader
                step_count += 1
                val, grads = Flux.withgradient(m -> learner.loss_fn(m(x), y), learner.model)
                Flux.update!(learner.optim, learner.model, grads[1])

                computed_quantities = Dict{Symbol,Float32}()
                for q in learner.quantities
                    value = compute!(q, val, grads[1])
                    computed_quantities[quantity_key(q)] = value
                end
                put!(channel, (step_count, computed_quantities))
                sleep(0.001) # To make concurrency possible
            end
            @info "Epoch $epoch complete"
        end
    catch e
        @error "Training Error" exception = (e, catch_backtrace())
    finally
        close(channel)
    end
end
