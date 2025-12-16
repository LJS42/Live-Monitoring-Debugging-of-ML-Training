using Flux
using Logging

"""
    Learner

Object bundling together all information for training with cockpit. Mutable struct so that model parameters and optimizer state
can be updated in-place during training.
Fields:
    model: Any
        arquitecture which parameters should be optimised during training
    data_loader: Any
        iterable that makes data batches accesible for training
    loss_fn: Function
        calculate the loss of the model w.t.r to training objective (returns scalar loss value)
    optim: Any
        optimizer chosen to update model parameters in the backward pass (e.g. Flux.Adam)
    quantities: Vector{<:AbstractQuantity}
        (optional) metrics computed every training step used for evakuation and diagnostic of model training
"""
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

"""
    Train!(learner, epochs, with_plots)

train a Learner and render quantities (optinal).
when plotting desired: create channel to pass data from training loop to cockpit session:
    - use put! to pass data into the channel (if full wait until space availiable, else add inmediately),
    - use take! to returm data from the channel (if channel is empty wait until data arrives, else retrieve inmediately).

Args:
    learner: Learner
        contains model and model training specifications (architecture, loss function, optimizer, quantities to track, etc.)
    epochs: Int
        number of training epochs
    with_plots: Bool
        user selection, if rendering is desired with_plots = True
"""
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
