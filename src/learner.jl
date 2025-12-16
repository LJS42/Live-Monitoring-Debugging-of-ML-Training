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
#outer constructor to create a Learner object with a default empty quantity vector if the user does not pass in any quantities
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
    if with_plots
        ch = Channel{Tuple{Int,Dict{Symbol,Float32}}}(100)

        train_task = @async train_loop!(learner, epochs, ch)
        render_task = @async render_loop(ch, learner.quantities)

        wait(train_task)
        wait(render_task)
    else 
        train_task = @async train_loop!(learner, epochs, nothing)
        wait(train_task)
    end
    
end

"""
    train_loop!(learner, epochs, channel)

Run training for a Learner and send training quantities through a channel for visualization 
Perform model optimization loop: iteration over epochs and batches, use loss and corresponding gradients w.r.t trainable 
parameters to update the model in-place and compute optinal metrics (quantities).
Use a global step counter for traning steps and a channel that automatically closes if task is finished or an error occurs

Args: 
    learner: Learner,
        contains model and model training specifications (architecture, loss function, optimizer, quantities to track, etc.)
    epochs: Int,
        number of training epochs
    channel: Channel{Tuple{Int,Dict{Symbol,Float32}}} or nothing
        communication channel with capacity set to 100 to pass information between Flux backend and cockpit, needed for plotting

"""
function train_loop!(
    learner::Learner,
    epochs::Int,
    channel::Union{Channel{Tuple{Int,Dict{Symbol,Float32}}}, Nothing}
)
    step_count = 0
    # coverage: ignore start
    try
        for epoch in 1:epochs
            for (x, y) in learner.data_loader
                step_count += 1
                val, grads = Flux.withgradient(m -> learner.loss_fn(m(x), y), learner.model)
                Flux.update!(learner.optim, learner.model, grads[1])
                
                if channel !== nothing
                    computed_quantities = Dict{Symbol,Float32}()
                    for q in learner.quantities
                        value = compute!(q, val, grads[1])
                        computed_quantities[quantity_key(q)] = value
                    end
                    put!(channel, (step_count, computed_quantities))
                    sleep(0.001) # To make concurrency possible
                end
            end
            @info "Epoch $epoch complete"
        end
    catch e
        @error "Training Error" exception = (e, catch_backtrace())
    finally
        if channel !== nothing
            close(channel)
        end
    end
    # coverage: ignore end
end
