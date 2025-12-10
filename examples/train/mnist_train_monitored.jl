using Statistics
using Flux
using MLDatasets
using WGLMakie
using Bonito
using Dates
using Base.Threads

# Data Loader
function get_data()
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

function main()
    # Thread check
    if Threads.nthreads() == 1
        @warn "Running on 1 thread. For best stability, restart with: julia --threads=auto ..."
    else
        @info "Running with $(Threads.nthreads()) threads."
    end

    set_theme!(theme_black())
    fig = Figure(size=(1000, 600), fontsize=18)

    ax = Axis(fig[1, 1],
        title="Training",
        xlabel="Step", ylabel="Loss (Log10)",
        yscale=log10
    )

    # Initial limits
    xlims!(ax, 0, 100)
    ylims!(ax, 0.01, 10.0)

    obs_batch_loss = Observable(Point2f[])
    obs_avg_loss = Observable(Point2f[])

    lines!(ax, obs_batch_loss, color=(:cyan, 0.3), linewidth=1, label="Batch Loss")
    lines!(ax, obs_avg_loss, color=:orange, linewidth=3, label="Moving Avg")
    axislegend(ax)

    # Setup browser session for WGLMakie
    WGLMakie.activate!()
    Bonito.browser_display()
    display(fig)

    # This is used to collect metrics asynchronously
    data_channel = Channel{Tuple{Int,Float32,Float32}}(1000)
    is_training = Observable(true)

    # Render Loop
    fps = 30
    plot_data_batch = Point2f[]
    plot_data_avg = Point2f[]
    current_xmax = 100.0

    Timer(1 / fps; interval=1 / fps) do t
        if !is_training[] && !isready(data_channel)
            close(t)
            return
        end

        has_new_data = false
        last_step = 0

        while isready(data_channel)
            step, val, avg = take!(data_channel)
            push!(plot_data_batch, Point2f(step, val))
            push!(plot_data_avg, Point2f(step, avg))
            last_step = step
            has_new_data = true
        end

        if has_new_data
            obs_batch_loss[] = plot_data_batch
            obs_avg_loss[] = plot_data_avg

            # Scale axis only if we need to
            if last_step > current_xmax
                new_xmax = current_xmax * 1.5
                xlims!(ax, 0, new_xmax)
                current_xmax = new_xmax
            end
        end
    end

    # Training Loop
    @info "Starting Training..."
    function training_job()
        try
            train_loader = get_data()
            model = get_model()
            loss_fn(ŷ, y) = Flux.logitcrossentropy(ŷ, y)
            optim = Flux.setup(Adam(3.0f-4), model)

            step_count = 0
            loss_buffer = Float32[]

            for epoch in 1:5
                for (x, y) in train_loader
                    step_count += 1

                    val, grads = Flux.withgradient(m -> loss_fn(m(x), y), model)
                    Flux.update!(optim, model, grads[1])
                    push!(loss_buffer, val)

                    if step_count % 5 == 0
                        avg = mean(loss_buffer[max(1, end - 50):end])
                        put!(data_channel, (step_count, val, avg))
                        yield() # Julias scheduler seems to require this to avoid blocking
                    end
                end
                @info "Epoch $epoch complete"
            end
        catch e
            @error "Training Error" exception = (e, catch_backtrace())
        finally
            is_training[] = false
            close(data_channel)
        end
    end

    # If we have multiple threads, launch the training job in a separate thread,
    # otherwise use concurrency
    local task
    if Threads.nthreads() > 1
        task = Threads.@spawn training_job()
    else
        task = @async training_job()
    end

    # Wait until training is done
    wait(task)

    @info "Training complete"
end

main()

# So Flux seems to provide the AbstractMetric type
# and will call these methods for every configured metric
#
# Called once before training starts.
# function setup_viz!(metric::AbstractMetric, layout_slot)
# ...
# end

# Called every batch.
# 'state' contains {model, x, y, y_pred, loss, grads, step_index}
# function on_batch_end!(metric::AbstractMetric, state)
# ...
# end

# Called every epoch.
# 'state' contains {model, val_loader, epoch_index}
# function on_epoch_end!(metric::AbstractMetric, state)
# ...
# end
#
# I think we can just build upon this type, extract the data, push it through channels,
# compute, and then visualize (similar to the render loop here).
# It probably would make sense to define data structures that will go through these channels per-metric as well.
# From the project structure it also seems to me like we want the decouple the data/compute and visualize part?
# So for example that we can visualize the same metric in different ways by just swapping the type the data gets fed into?
